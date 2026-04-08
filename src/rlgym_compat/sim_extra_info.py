from collections import deque
from typing import Dict, Optional

import RocketSim as rsim
from rlbot.flat import (
    BoostStrengthMutator,
    DemolishMutator,
    FieldInfo,
    GameMode,
    GamePacket,
    GravityMutator,
    MatchConfiguration,
    PlayerInfo,
    RespawnTimeMutator
)

from .car import Car
from .extra_info import ExtraBallInfo, ExtraPacketInfo, ExtraPlayerInfo
from .math import euler_to_rotation
from .utils import rotator_to_numpy, vector_to_numpy


class SimExtraInfo:
    def __init__(
        self, field_info: FieldInfo, match_settings: MatchConfiguration, tick_skip: int = 8
    ):
        # Determine Game Mode - v5 uses string or enum comparison
        match_mode = match_settings.game_mode
        if match_mode == 1: # Hoops
            mode = rsim.GameMode.HOOPS
        elif match_mode == 3: # Snowday
            mode = rsim.GameMode.SNOWDAY
        else:
            mode = rsim.GameMode.SOCCAR 

        mutators = match_settings.mutators
        mutator_config = {}
        
        if mutators:
            # Gravity: default -650.0
            grav_z = -650.0
            if mutators.gravity == GravityMutator.Low: grav_z = -325.0
            elif mutators.gravity == GravityMutator.High: grav_z = -1137.5
            elif mutators.gravity == GravityMutator.SuperHigh: grav_z = -3250.0
            elif mutators.gravity == GravityMutator.Reverse: grav_z = 650.0
            mutator_config["gravity"] = rsim.Vec(0, 0, grav_z)

            # Boost Strength
            if mutators.boost_strength == BoostStrengthMutator.OneAndAHalf: 
                mutator_config["boost_accel"] = 21.2 * 1.5
            elif mutators.boost_strength == BoostStrengthMutator.Two: 
                mutator_config["boost_accel"] = 21.2 * 2
            elif mutators.boost_strength == BoostStrengthMutator.Five: 
                mutator_config["boost_accel"] = 21.2 * 5
            elif mutators.boost_strength == BoostStrengthMutator.Ten: 
                mutator_config["boost_accel"] = 21.2 * 10

        # Simulation Setup
        self._arena = rsim.Arena(mode)
        self._arena.set_ball_touch_callback(self._ball_touch_callback)
        if mutators:
            self._arena.set_mutator_config(rsim.MutatorConfig(**mutator_config))

        self._ball_touched_on_tick: Dict[int, bool] = {}
        self._touches: Dict[int, deque[bool]] = {}
        
        # Identity Mappings (Index is the only stable ID in v5)
        self._index_to_car_id: Dict[int, int] = {}
        self._car_id_to_index: Dict[int, int] = {}
        
        self._tick_skip = tick_skip
        self._first_call = True
        self._tick_count = 0

    def _ball_touch_callback(self, arena: rsim.Arena, car: rsim.Car, data):
        self._ball_touched_on_tick[car.id] = True

    def _get_extra_ball_info(self) -> ExtraBallInfo:
        ball_state = self._arena.ball.get_state()
        return ExtraBallInfo(
            heatseeker_target_dir=ball_state.heatseeker_target_dir,
            heatseeker_target_speed=ball_state.heatseeker_target_speed,
            heatseeker_time_since_hit=ball_state.heatseeker_time_since_hit,
        )

    def _get_extra_player_info(self, car: rsim.Car) -> ExtraPlayerInfo:
        car_state = car.get_state()
        contact_index = -1
        if car_state.car_contact_id != 0:
            contact_index = self._car_id_to_index.get(car_state.car_contact_id, -1)

        return ExtraPlayerInfo(
            on_ground=car_state.is_on_ground,
            handbrake=car_state.handbrake_val,
            ball_touches=sum(self._touches.get(car.id, [False])),
            car_contact_id=contact_index,
            car_contact_cooldown_timer=car_state.car_contact_cooldown_timer,
            is_autoflipping=car_state.is_auto_flipping,
            autoflip_timer=car_state.auto_flip_timer,
            autoflip_direction=car_state.auto_flip_torque_scale,
        )

    def _get_extra_packet_info(self) -> ExtraPacketInfo:
        sim_cars = self._arena.get_cars()
        # Ensure we return info in the same order as the packet players
        sorted_cars = sorted(sim_cars, key=lambda c: self._car_id_to_index.get(c.id, 999))
        players = [self._get_extra_player_info(car) for car in sorted_cars]
        return ExtraPacketInfo(players=players, ball=self._get_extra_ball_info())

    def get_extra_info(self, packet: GamePacket) -> ExtraPacketInfo:
        # Re-initialize if player count changes
        if len(packet.players) != len(self._index_to_car_id):
            self._rebuild_cars(packet)

        if self._first_call:
            self._first_call = False
            self._tick_count = packet.match_info.frame_num
            self._sync_all_states(packet)
            return self._get_extra_packet_info()

        ticks_elapsed = packet.match_info.frame_num - self._tick_count
        self._tick_count = packet.match_info.frame_num

        # Apply Inputs
        for i, player in enumerate(packet.players):
            car_id = self._index_to_car_id.get(i)
            if car_id is not None:
                car = self._arena.get_car_from_id(car_id)
                ctrl = rsim.CarControls()
                inp = player.last_input
                if inp:
                    ctrl.throttle, ctrl.steer = inp.throttle, inp.steer
                    ctrl.pitch, ctrl.yaw, ctrl.roll = inp.pitch, inp.yaw, inp.roll
                    ctrl.boost, ctrl.jump, ctrl.handbrake = inp.boost, inp.jump, inp.handbrake
                    car.set_controls(ctrl)

        # Step Sim
        for _ in range(max(0, ticks_elapsed)):
            for cid in self._car_id_to_index.keys():
                self._ball_touched_on_tick[cid] = False
            
            self._arena.step(1)
            
            for cid in self._car_id_to_index.keys():
                self._touches[cid].append(self._ball_touched_on_tick.get(cid, False))

        self._sync_all_states(packet)
        return self._get_extra_packet_info()

    def _rebuild_cars(self, packet: GamePacket):
        for car in self._arena.get_cars():
            self._arena.remove_car(car.id)
        
        self._index_to_car_id.clear()
        self._car_id_to_index.clear()
        self._touches.clear()

        for i, player in enumerate(packet.players):
            # v5 hitbox access: player.hitbox and player.hitbox_offset
            # Car.detect_hitbox must handle these correctly
            hitbox_data = player.hitbox
            offset_data = player.hitbox_offset
            
            config = Car.detect_hitbox(hitbox_data, offset_data)
            sim_car = self._arena.add_car(player.team, config)
            
            self._index_to_car_id[i] = sim_car.id
            self._car_id_to_index[sim_car.id] = i
            self._touches[sim_car.id] = deque([False] * self._tick_skip, maxlen=self._tick_skip)

    def _sync_all_states(self, packet: GamePacket):
        # Ball Sync
        if len(packet.balls) > 0:
            ball = self._arena.ball
            p_ball = packet.balls[0].physics
            b_state = ball.get_state()
            
            # Explicitly cast to floats/tuples for rsim.Vec
            loc = p_ball.location
            vel = p_ball.velocity
            ang = p_ball.angular_velocity
            
            b_state.pos = rsim.Vec(loc.x, loc.y, loc.z)
            b_state.vel = rsim.Vec(vel.x, vel.y, vel.z)
            b_state.ang_vel = rsim.Vec(ang.x, ang.y, ang.z)
            
            rot_np = rotator_to_numpy(p_ball.rotation)
            b_state.rot_mat = rsim.RotMat(*euler_to_rotation(rot_np).transpose().flatten())
            ball.set_state(b_state)

        # Car Sync
        for i, player in enumerate(packet.players):
            car_id = self._index_to_car_id.get(i)
            if car_id is not None:
                car = self._arena.get_car_from_id(car_id)
                c_state = car.get_state()
                phys = player.physics
                
                loc, vel, ang = phys.location, phys.velocity, phys.angular_velocity
                c_state.pos = rsim.Vec(loc.x, loc.y, loc.z)
                c_state.vel = rsim.Vec(vel.x, vel.y, vel.z)
                c_state.ang_vel = rsim.Vec(ang.x, ang.y, ang.z)
                
                rot_np = rotator_to_numpy(phys.rotation)
                c_state.rot_mat = rsim.RotMat(*euler_to_rotation(rot_np).transpose().flatten())
                c_state.boost = player.boost
                c_state.is_demoed = player.demolished_timeout != -1
                car.set_state(c_state)