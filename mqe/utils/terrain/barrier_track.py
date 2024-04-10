import numpy as np
import torch
from copy import copy

from isaacgym import gymapi, gymutil
from isaacgym.terrain_utils import convert_heightfield_to_trimesh
from mqe.utils import trimesh
from mqe.utils.terrain.perlin import TerrainPerlin
from mqe.utils.console import colorize

class BarrierTrack:
    # default kwargs
    track_kwargs = dict(
            options = [
                "gate",
                "init",
                "wall",
                "plane",
            ],
            track_width = 1.6,
            track_length = None,
            # track_block_length = 1.2, # the x-axis distance from the env origin point
            wall_thickness = 0.04, # [m]
            wall_height = 0.5, # [m]
            wall = dict(
                block_length = 3.0,
            ),
            plane = dict(
                block_length = 3.0,
            ),
            init = dict(
                block_length = 1.2,
                room_size = (0.8, 0.8),
                border_with = 0.05,
                offset = (0, 0),
            ),
            gate = dict(
                block_length = 1.2,
                width = 1.,
                depth = 1., # size along the forward axis
                offset = (0, 0),
            ),
            # If True, will add perlin noise to each surface which will be step on. And please
            # provide self.cfg.TerrainPerlin_kwargs for generating Perlin noise
            add_perlin_noise = False,
            border_perlin_noise = False,
            border_height = 0., # Incase we want the surrounding plane to be lower than the track
            virtual_terrain = False,
            check_skill_combinations = False, # check if some specific skills are connected, if set. e.g. climb -> leap
            engaging_next_threshold = 0., # if > 0, engaging_next is based on this threshold instead of track_block_length/2. Make sure the obstacle is not too long.
            curriculum_perlin = True, # If True, perlin noise scale will be depends on the difficulty if possible.
            no_perlin_threshold = 0.02, # If the perlin noise is too small, clip it to zero.
        )
    
    def __init__(self, cfg, num_envs: int, num_agents=1) -> None:
        self.cfg = cfg
        self.num_envs = num_envs
        
        assert self.cfg.mesh_type == "trimesh", "Not implemented for mesh_type other than trimesh, get {}".format(self.cfg.mesh_type)
        assert getattr(self.cfg, "BarrierTrack_kwargs", None) is not None, "Must provide BarrierTrack_kwargs in cfg.terrain"

        self.track_kwargs.update(self.cfg.BarrierTrack_kwargs)
        if self.track_kwargs["add_perlin_noise"] and not hasattr(self.cfg, "TerrainPerlin_kwargs"):
            print(colorize(
                "Warning: Please provide cfg.terrain.TerrainPerlin to configure perlin noise for all surface to step on.",
                color= "yellow",
            ))

        self.num_agents = num_agents
        self.env_origins = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 3), dtype= np.float32)
        self.agent_origins = np.zeros((self.cfg.num_rows, self.cfg.num_cols, self.num_agents, 3), dtype= np.float32)

    def initialize_track_info_buffer(self):
        """ Build buffers to store oracle info for each track blocks so that it is faster to compute
        oracle observation. Some dimensions are predefined.
        """
        # For each track block (n_options), 3 parameters are enabled:
        # - track_id: int, starting track is 0, other numbers depends on the options order.
        # - obstacle_depth: float,
        # - obstacle_critical_params: e.g. tilt width, crawl height, climb height

        # num_rows + 1 incase the robot finish the entire row of tracks

        self.track_width_map = torch.zeros(
            (self.cfg.num_rows, self.cfg.num_cols),
            dtype= torch.float32,
            device= self.device,
        )

    def initialize_track(self):
        """ A track is defined as follow

              |-----------track length-----------|  
                        |-block length-|
          +y^ +---------+--------------+---------+   --
            | |         |              |         |    |
            | | block 0 |    block 1   | block 2 |    | track width 
            | |         |              |         |    |
            | +---------+--------------+---------+   --
            +--------->
                     +x
        
        """
        # add block length up to get track length
        track_length = 0.
        self.env_block_lengths = []
        self.track_block_resolutions = []
        for option in self.track_kwargs["options"]:
            track_length += self.track_kwargs[option]["block_length"]
            self.env_block_lengths.append(self.track_kwargs[option]["block_length"])
            self.track_block_resolutions.append((
                np.ceil(self.track_kwargs[option]["block_length"] / self.cfg.horizontal_scale).astype(int),
                np.ceil(self.track_kwargs["track_width"] / self.cfg.horizontal_scale).astype(int),
            ))
        self.track_kwargs["track_length"] = track_length
        self.track_resolution = (
            np.ceil(self.track_kwargs["track_length"] / self.cfg.horizontal_scale).astype(int),
            np.ceil(self.track_kwargs["track_width"] / self.cfg.horizontal_scale).astype(int),
        )
        self.n_blocks_per_track = len(self.track_kwargs["options"])
        self.env_length = track_length
        self.env_width = self.track_kwargs["track_width"]

    ##### methods that generate models and critical parameters of each track block #####
    def fill_heightfield_to_scale(self, heightfield):
        """ Due to the rasterization of the heightfield, the trimesh size does not match the 
        heightfield_resolution * horizontal_scale, so we need to fill enlarge heightfield to
        meet this scale.
        """
        assert len(heightfield.shape) == 2, "heightfield must be 2D"
        heightfield_x_fill = np.concatenate([
            heightfield,
            heightfield[-2:, :],
        ], axis= 0)
        heightfield_y_fill = np.concatenate([
            heightfield_x_fill,
            heightfield_x_fill[:, -2:],
        ], axis= 1)
        return heightfield_y_fill

    """ Example block
        
           |-----block length------|
        
           +-----------------------+   ---
           |                       |     |
           |                       |     |
      +y^  |         block         |     | track width
        |  |                       |     |
        |  |                       |     |
        |  +-----------------------+   ---
        +----------->
                    +x

    """

    def get_wall_block(self,
            wall_thickness,
            block_resolution,
            difficulty = None,
            block_origin_px = None
        ):
        block_heighfield = np.zeros(block_resolution, dtype= np.float32)
        heighfield_noise_mask = np.zeros(block_resolution, dtype= np.float32)
        reset_pos_px = None
        
        wall_height = ( \
            np.random.uniform(*self.track_kwargs["wall_height"]) \
            if isinstance(self.track_kwargs["wall_height"], (tuple, list)) \
            else self.track_kwargs["wall_height"] \
        ) / self.cfg.vertical_scale

        block_heighfield[ :, :] = wall_height

        block_info = {}
        
        height_offset_px = 0

        return block_heighfield, block_info, heighfield_noise_mask, height_offset_px, reset_pos_px

    def get_plane_block(self,
            wall_thickness,
            block_resolution,
            difficulty = None,
            block_origin_px = None
        ):
        block_heighfield = np.zeros(block_resolution, dtype= np.float32)
        heighfield_noise_mask = np.zeros(block_resolution, dtype= np.float32)
        reset_pos_px = None
        
        wall_height = ( \
            np.random.uniform(*self.track_kwargs["wall_height"]) \
            if isinstance(self.track_kwargs["wall_height"], (tuple, list)) \
            else self.track_kwargs["wall_height"] \
        ) / self.cfg.vertical_scale
        wall_thickness_px = np.ceil(wall_thickness / self.cfg.horizontal_scale).astype(int)

        block_heighfield[ :, : wall_thickness_px] = wall_height
        block_heighfield[ :, -wall_thickness_px :] = wall_height
        heighfield_noise_mask[ :, wall_thickness_px: block_resolution[1] - wall_thickness_px] = 1.

        block_info = {}
        
        height_offset_px = 0

        return block_heighfield, block_info, heighfield_noise_mask, height_offset_px, reset_pos_px

    def get_init_block(self,
            wall_thickness,
            block_resolution,
            difficulty = None,
            block_origin_px = None
        ):
        block_heighfield = np.zeros(block_resolution, dtype= np.float32)
        heighfield_noise_mask = np.zeros(block_resolution, dtype= np.float32)
        reset_pos_px = np.zeros((self.num_agents, 3), dtype= np.float32)
        
        wall_height = ( \
            np.random.uniform(*self.track_kwargs["wall_height"]) \
            if isinstance(self.track_kwargs["wall_height"], (tuple, list)) \
            else self.track_kwargs["wall_height"] \
        ) / self.cfg.vertical_scale

        room_offset_px = (int(self.track_kwargs["init"]["offset"][0] / self.cfg.horizontal_scale), \
                          int(self.track_kwargs["init"]["offset"][1] / self.cfg.horizontal_scale))
        room_size_px = (int(self.track_kwargs["init"]["room_size"][0] / self.cfg.horizontal_scale), \
                        int(self.track_kwargs["init"]["room_size"][1] / self.cfg.horizontal_scale))
        border_px = np.ceil(self.track_kwargs["init"]["border_width"] / self.cfg.horizontal_scale).astype(int)
        wall_thickness_px = np.ceil(wall_thickness / self.cfg.horizontal_scale).astype(int)

        # make room for agents
        room_xsize_px = room_size_px[0]
        room_ysize_px = room_size_px[1] * self.num_agents + border_px * (self.num_agents - 1)
        room_origin_px = (np.ceil((block_resolution[0] - room_xsize_px) / 2).astype(int) + room_offset_px[0], \
                          np.ceil((block_resolution[1] - room_ysize_px) / 2).astype(int) + room_offset_px[1])
        
        block_heighfield[ : room_origin_px[0] + room_xsize_px, :] = wall_height
        block_heighfield[ :, : wall_thickness_px] = wall_height
        block_heighfield[ :, -wall_thickness_px :] = wall_height
        heighfield_noise_mask[room_origin_px[0] + room_xsize_px :, wall_thickness_px: block_resolution[1] - wall_thickness_px] = 1.

        for i in range(self.num_agents):
            block_heighfield[
                room_origin_px[0] : room_origin_px[0] + room_size_px[0], \
                room_origin_px[1] + i *  (room_size_px[1] + border_px): room_origin_px[1] + (i + 1) *  room_size_px[1] + i * border_px
            ] = 0.
            heighfield_noise_mask[
                room_origin_px[0] : room_origin_px[0] + room_size_px[0], \
                room_origin_px[1] + i *  (room_size_px[1] + border_px): room_origin_px[1] + (i + 1) *  room_size_px[1] + i * border_px
            ] = 1.
            reset_pos_px[i, 0] = room_origin_px[0] + int(room_size_px[0] / 2)
            reset_pos_px[i, 1] = room_origin_px[1] + i *  (room_size_px[1] + border_px) + int(room_size_px[1] / 2)

        block_heighfield[ :, : wall_thickness_px] = wall_height
        block_heighfield[ :, -wall_thickness_px :] = wall_height
        block_heighfield[ : wall_thickness_px, :] = wall_height

        block_info = {}
        
        height_offset_px = 0

        return block_heighfield, block_info, heighfield_noise_mask, height_offset_px, reset_pos_px

    def get_rotation_block(self,
            wall_thickness,
            block_resolution,
            difficulty = None,
            virtual = False,
            block_origin_px = None
        ):
        block_heighfield = np.zeros(block_resolution, dtype= np.float32)
        heighfield_noise_mask = np.ones(block_resolution, dtype= np.float32)
        reset_pos_px = None
        
        rotation_depth = np.random.uniform(*self.track_kwargs["rotation"]["depth"]) \
                        if isinstance(self.track_kwargs["rotation"]["depth"], (tuple, list)) \
                        else self.track_kwargs["rotation"]["depth"]
        wall_height = np.random.uniform(*self.track_kwargs["wall_height"]) \
                                if isinstance(self.track_kwargs["wall_height"], (tuple, list)) \
                                else self.track_kwargs["wall_height"]
        offset_px = (np.ceil(self.track_kwargs["rotation"]["offset"][0] / self.cfg.horizontal_scale).astype(int), \
                     np.ceil(self.track_kwargs["rotation"]["offset"][1] / self.cfg.horizontal_scale).astype(int))
        
        wide_px = (np.ceil(self.track_kwargs["rotation"]["wide_px"][0] / self.cfg.horizontal_scale).astype(int), \
                     np.ceil(self.track_kwargs["rotation"]["wide_px"][1] / self.cfg.horizontal_scale).astype(int))
        
        depth_px = int(rotation_depth / self.cfg.horizontal_scale)
        height_value = wall_height / self.cfg.vertical_scale
        wall_thickness_px = np.ceil(wall_thickness / self.cfg.horizontal_scale).astype(int)

        rotation_origin = np.asarray([np.ceil((block_resolution[0] - depth_px) / 2).astype(int) + offset_px[0], \
                                    np.ceil((block_resolution[1] - depth_px) / 2).astype(int) + offset_px[1]])

        block_heighfield[rotation_origin[0] : rotation_origin[0] + depth_px, : wide_px[0]] = height_value
        block_heighfield[rotation_origin[0] : rotation_origin[0] + depth_px, -wide_px[0] :] = height_value
        block_heighfield[ :, : wall_thickness_px] = height_value
        block_heighfield[ :, -wall_thickness_px :] = height_value

        heighfield_noise_mask[rotation_origin[0] : rotation_origin[0] + depth_px, : wide_px[0]] = 0.
        heighfield_noise_mask[rotation_origin[0] : rotation_origin[0] + depth_px, -wide_px[0] :] = 0.
        heighfield_noise_mask[ :, : wall_thickness_px] = 0.
        heighfield_noise_mask[ :, -wall_thickness_px :] = 0.

        block_info = {
            "rotation_size": torch.tensor(rotation_depth, dtype= torch.float32, device= self.device) * self.cfg.horizontal_scale,
        }
        height_offset_px = 0
        
        return block_heighfield, block_info, heighfield_noise_mask, height_offset_px, reset_pos_px

    def get_gate_block(self,
            wall_thickness,
            block_resolution,
            difficulty = None,
            block_origin_px = None
        ):
        block_heighfield = np.zeros(block_resolution, dtype= np.float32)
        heighfield_noise_mask = np.ones(block_resolution, dtype= np.float32)
        reset_pos_px = None
        
        gate_depth = np.random.uniform(*self.track_kwargs["gate"]["depth"]) \
                        if isinstance(self.track_kwargs["gate"]["depth"], (tuple, list)) \
                        else self.track_kwargs["gate"]["depth"]
        wall_height = np.random.uniform(*self.track_kwargs["wall_height"]) \
                                if isinstance(self.track_kwargs["wall_height"], (tuple, list)) \
                                else self.track_kwargs["wall_height"]
        offset_px = np.asarray((np.ceil(self.track_kwargs["gate"]["offset"][0] / self.cfg.horizontal_scale).astype(int), \
                                np.ceil(self.track_kwargs["gate"]["offset"][1] / self.cfg.horizontal_scale).astype(int)))
        random_px = np.asarray((self.track_kwargs["gate"]["random"][0] / self.cfg.horizontal_scale, \
                                self.track_kwargs["gate"]["random"][1] / self.cfg.horizontal_scale))
        random_px = np.ceil(random_px * (np.random.random(2) - 0.5) * 2).astype(int)
        
        if isinstance(self.track_kwargs["gate"]["width"], (tuple, list)):
            if difficulty is None:
                gate_width = np.random.uniform(*self.track_kwargs["gate"]["width"])
            else:
                gate_width = difficulty * self.track_kwargs["gate"]["width"][0] + (1-difficulty) * self.track_kwargs["gate"]["width"][1]
        else:
            gate_width = self.track_kwargs["gate"]["width"]
        depth_px = int(gate_depth / self.cfg.horizontal_scale)
        width_px = int(gate_width / self.cfg.horizontal_scale)
        height_value = wall_height / self.cfg.vertical_scale
        wall_thickness_px = np.ceil(wall_thickness / self.cfg.horizontal_scale).astype(int)
        gate_origin = np.asarray([np.ceil((block_resolution[0] - depth_px) / 2).astype(int), \
                                  np.ceil((block_resolution[1] - width_px) / 2).astype(int)]) + offset_px + random_px

        block_heighfield[gate_origin[0] : gate_origin[0] + depth_px, :] = height_value
        block_heighfield[ :, : wall_thickness_px] = height_value
        block_heighfield[ :, -wall_thickness_px :] = height_value
        heighfield_noise_mask[gate_origin[0] : gate_origin[0] + depth_px, :] = 0.
        heighfield_noise_mask[ :, : wall_thickness_px] = 0.
        heighfield_noise_mask[ :, -wall_thickness_px :] = 0.
        block_heighfield[gate_origin[0] : gate_origin[0] + depth_px, gate_origin[1] : gate_origin[1] + width_px] = 0.
        heighfield_noise_mask[gate_origin[0] : gate_origin[0] + depth_px, gate_origin[1] : gate_origin[1] + width_px] = 1.

        block_info = {
            "gate_deviation": torch.tensor(offset_px + random_px, dtype= torch.float32, device= self.device) * self.cfg.horizontal_scale,
        }
        
        height_offset_px = 0
        
        return block_heighfield, block_info, heighfield_noise_mask, height_offset_px, reset_pos_px

    ##### initialize and building tracks then entire terrain #####
    def build_heightfield_raw(self):
        self.border = int(self.cfg.border_size / self.cfg.horizontal_scale)
        map_x_size = int(self.cfg.num_rows * self.track_resolution[0]) + 2 * self.border
        map_y_size = int(self.cfg.num_cols * self.track_resolution[1]) + 2 * self.border
        self.tot_rows = map_x_size # TODO: change name to map_x_px
        self.tot_cols = map_y_size
        self.heightfield_raw = np.zeros((map_x_size, map_y_size), dtype= np.float32)
        if self.track_kwargs["add_perlin_noise"] and self.track_kwargs["border_perlin_noise"]:
            TerrainPerlin_kwargs = self.cfg.TerrainPerlin_kwargs
            for k, v in self.cfg.TerrainPerlin_kwargs.items():
                if isinstance(v, (tuple, list)):
                    if TerrainPerlin_kwargs is self.cfg.TerrainPerlin_kwargs:
                        TerrainPerlin_kwargs = copy(self.cfg.TerrainPerlin_kwargs)
                    TerrainPerlin_kwargs[k] = v[0]
            heightfield_noise = TerrainPerlin.generate_fractal_noise_2d(
                xSize= self.env_length * self.cfg.num_rows + 2 * self.cfg.border_size,
                ySize= self.env_width * self.cfg.num_cols + 2 * self.cfg.border_size,
                xSamples= map_x_size,
                ySamples= map_y_size,
                **TerrainPerlin_kwargs,
            ) / self.cfg.vertical_scale
            self.heightfield_raw += heightfield_noise
            # self.heightfield_raw[self.border:-self.border, self.border:-self.border] = 0.
            if self.track_kwargs["border_height"] != 0.:
                # self.heightfield_raw[:self.border, :] += self.track_kwargs["border_height"] / self.cfg.vertical_scale
                # self.heightfield_raw[-self.border:, :] += self.track_kwargs["border_height"] / self.cfg.vertical_scale
                self.heightfield_raw[:, :self.border] += self.track_kwargs["border_height"] / self.cfg.vertical_scale
                self.heightfield_raw[:, -self.border:] += self.track_kwargs["border_height"] / self.cfg.vertical_scale
        self.heightsamples = self.heightfield_raw

    def add_trimesh_to_sim(self, trimesh, trimesh_origin):
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = trimesh[0].shape[0]
        tm_params.nb_triangles = trimesh[1].shape[0]
        tm_params.transform.p.x = trimesh_origin[0]
        tm_params.transform.p.y = trimesh_origin[1]
        tm_params.transform.p.z = trimesh_origin[2]
        tm_params.static_friction = self.cfg.static_friction
        tm_params.dynamic_friction = self.cfg.dynamic_friction
        tm_params.restitution = self.cfg.restitution
        self.gym.add_triangle_mesh(
            self.sim,
            trimesh[0].flatten(order= "C"),
            trimesh[1].flatten(order= "C"),
            tm_params,
        )

    def add_track_to_sim(self, track_origin_px, row_idx= None, col_idx= None):
        """ add heighfield value and add trimesh to sim for one certain race track """
        # adding trimesh and heighfields
        block_order = self.track_kwargs["options"].copy()
        difficulties = self.get_difficulty(row_idx, col_idx)
        difficulty, virtual_track = difficulties[:2]
        env_origins = None
        track_info = {}

        if self.track_kwargs["add_perlin_noise"]:
            TerrainPerlin_kwargs = self.cfg.TerrainPerlin_kwargs
            for k, v in self.cfg.TerrainPerlin_kwargs.items():
                if isinstance(v, (tuple, list)):
                    if TerrainPerlin_kwargs is self.cfg.TerrainPerlin_kwargs:
                        TerrainPerlin_kwargs = copy(self.cfg.TerrainPerlin_kwargs)
                    if difficulty is None or (not self.track_kwargs["curriculum_perlin"]):
                        TerrainPerlin_kwargs[k] = np.random.uniform(*v)
                    else:
                        TerrainPerlin_kwargs[k] = v[0] * (1 - difficulty) + v[1] * difficulty
                    if self.track_kwargs["no_perlin_threshold"] > TerrainPerlin_kwargs[k]:
                        TerrainPerlin_kwargs[k] = 0.
            heightfield_noise = TerrainPerlin.generate_fractal_noise_2d(
                xSize = self.env_length,
                ySize = self.env_width,
                xSamples = self.track_resolution[0],
                ySamples = self.track_resolution[1],
                **TerrainPerlin_kwargs,
            ) / self.cfg.vertical_scale

        # block_starting_height_px = track_origin_px[2]
        wall_thickness = np.random.uniform(*self.track_kwargs["wall_thickness"]) if isinstance(self.track_kwargs["wall_thickness"], (tuple, list)) else self.track_kwargs["wall_thickness"]
        
        block_origin_px = track_origin_px.copy()

        ### Creating Blocks Begin ###

        for block_idx, block_name in enumerate(block_order):

            block_heighfield, block_info, heighfield_noise_mask, height_offset_px, reset_pos = getattr(self, "get_" + block_name + "_block")( # i.e. get_init_block
                wall_thickness,
                self.track_block_resolutions[block_idx],
                block_origin_px = block_origin_px
            )

            self.heightfield_raw[
                block_origin_px[0]: block_origin_px[0] + self.track_block_resolutions[block_idx][0],
                block_origin_px[1]: block_origin_px[1] + self.track_block_resolutions[block_idx][1],
            ] = block_heighfield + heighfield_noise_mask * \
            self.heightfield_raw[
                block_origin_px[0]: block_origin_px[0] + self.track_block_resolutions[block_idx][0],
                block_origin_px[1]: block_origin_px[1] + self.track_block_resolutions[block_idx][1],
            ] + block_origin_px[2]

            block_origin_px[0] += self.track_block_resolutions[block_idx][0]
            block_origin_px[2] += height_offset_px

            # Record reset positions for agents
            if reset_pos is not None:
                if env_origins is not None:
                    print("Multiple reset block in the same track. Should be only one.")
                    exit()
                else:
                    env_origins = reset_pos
        
            for key in block_info.keys():
                track_info[key] = block_info[key]

        self.track_width_map[row_idx, col_idx] = self.env_width - wall_thickness * 2

        ### Creating Blocks End ###

        track_trimesh = convert_heightfield_to_trimesh(
            self.fill_heightfield_to_scale(self.heightfield_raw[
                track_origin_px[0]: track_origin_px[0] + self.track_resolution[0],
                track_origin_px[1]: track_origin_px[1] + self.track_resolution[1],
            ]),
            self.cfg.horizontal_scale,
            self.cfg.vertical_scale,
            self.cfg.slope_treshold,
        )
        self.add_trimesh_to_sim(track_trimesh,
            np.array([
                track_origin_px[0] * self.cfg.horizontal_scale,
                track_origin_px[1] * self.cfg.horizontal_scale,
                track_origin_px[2] * self.cfg.vertical_scale,
            ]))

        return env_origins, track_info

    def add_terrain_to_sim(self, gym, sim, device= "cpu"):
        """ Add current terrain as trimesh to sim and update the corresponding heightfield """
        self.gym = gym
        self.sim = sim
        self.device = device
        self.initialize_track()               # calculate size, resolution
        self.build_heightfield_raw()          # create border (height, perlin noise)
        self.initialize_track_info_buffer()

        """ The track grid is defined as follow
          +y^
            |         row 0         row 1
            |xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
            |xxxxxxxxxxxxxxx border xxxxxxxxxxxxxx
            |xxx +-------------+-------------+ xxx
            |xxx |    track    |    track    | xxx  col 2
            |xxx +-------------+-------------+ xxx
            |xxx |    track    |    track    | xxx  col 1
            |xxx +-------------+-------------+ xxx
            |xxx |    track    |    track    | xxx  col 0
            |xxx +-------------+-------------+ xxx
            |xxxxxxxxxxxxxxx border xxxxxxxxxxxxxx
            |xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
            +--------------------------------------->
                                                    +x

        """

        self.track_origins_px = np.zeros((self.cfg.num_rows, self.cfg.num_cols, 3), dtype= int)
        for col_idx in range(self.cfg.num_cols):
            starting_height_px = 0
            for row_idx in range(self.cfg.num_rows):
                self.track_origins_px[row_idx, col_idx] = [
                    int(row_idx * self.track_resolution[0]) + self.border,
                    int(col_idx * self.track_resolution[1]) + self.border,
                    starting_height_px,
                ]
                # NOTE: The starting heigh is passed to the `add_track_to_sim`.
                # The return value of `add_track_to_sim` is the z value, not the offset.
                env_origin, track_info = self.add_track_to_sim(
                    self.track_origins_px[row_idx, col_idx],
                    row_idx = row_idx,
                    col_idx = col_idx,
                )
                track_origin = self.track_origins_px[row_idx, col_idx].reshape(1, 3).repeat(self.num_agents, 0)
                self.agent_origins[row_idx, col_idx, :, :2] = (track_origin[:, :2] + env_origin[:, :2]) * self.cfg.horizontal_scale
                self.agent_origins[row_idx, col_idx, :, 2] = (track_origin[:, 2] + env_origin[:, 2]) * self.cfg.vertical_scale
        
                if getattr(self, "env_info", None) == None:
                    self.env_info = {}
                    for key in track_info.keys():
                        self.env_info[key] = track_info[key].reshape(1, 1, -1).repeat(self.cfg.num_rows, self.cfg.num_cols, 1)
                else:
                    for key in track_info.keys():
                        self.env_info[key][row_idx, col_idx, :] = track_info[key]

        self.add_plane_to_sim(starting_height_px)
        
        for i in range(self.cfg.num_rows):
            for j in range(self.cfg.num_cols):
                self.env_origins[i, j, 0] = self.track_origins_px[i, j, 0] * self.cfg.horizontal_scale
                self.env_origins[i, j, 1] = self.track_origins_px[i, j, 1] * self.cfg.horizontal_scale
                self.env_origins[i, j, 2] = self.track_origins_px[i, j, 2] * self.cfg.vertical_scale
                self.env_origins[i, j, 1] += self.track_kwargs["track_width"] / 2
        self.env_origins_pyt = torch.from_numpy(self.env_origins).to(self.device)

    def add_plane_to_sim(self, final_height_px= 0.):
        """
        Args:
            final_height_px: the height of the region 1 in the following figure.
        """
        if self.track_kwargs["add_perlin_noise"] and self.track_kwargs["border_perlin_noise"]:
            """
                 +----------------------------------------------------------------+
                 |                                                                |
                 |                                                                |
                 |region2                                                         |
                 +---------+-------------------------------------------+----------+
                 |         |                                           |          |
                 |         |                                           |          |
                 |         |                                           |          |
                 |         |                                           |          |
                 |         |                                           |          |
                 |         |              The track grid               |          |
                 |         |                                           |          |
                y|         |                                           |          |
                ^|         |                                           |          |
                ||region3  |                                           |   region1|
                |+---------+-------------------------------------------+----------+
                ||                                                                |
                ||                                                                |
                ||region0                                                         |
                |+----------------------------------------------------------------+
                +--------------->x
            """
            trimesh_origins = [
                [0, 0, 0],
                [self.cfg.border_size + self.cfg.num_rows * self.env_length, self.cfg.border_size, final_height_px * self.cfg.vertical_scale],
                [0, self.cfg.border_size + self.cfg.num_cols * self.env_width, 0],
                [0, self.cfg.border_size, 0],
            ]
            heightfield_regions = [
                [slice(0, self.heightfield_raw.shape[0]), slice(0, self.border)],
                [
                    slice(self.heightfield_raw.shape[0] - self.border, self.heightfield_raw.shape[0]),
                    slice(self.border, self.border + self.track_resolution[1] * self.cfg.num_cols),
                ],
                [
                    slice(0, self.heightfield_raw.shape[0]),
                    slice(self.heightfield_raw.shape[1] - self.border, self.heightfield_raw.shape[1]),
                ],
                [slice(0, self.border), slice(self.border, self.heightfield_raw.shape[1] - self.border)],
            ]
            for origin, heightfield_region in zip(trimesh_origins, heightfield_regions):
                plane_trimesh = convert_heightfield_to_trimesh(
                    self.fill_heightfield_to_scale(
                        self.heightfield_raw[
                            heightfield_region[0],
                            heightfield_region[1],
                        ]
                    ),
                    self.cfg.horizontal_scale,
                    self.cfg.vertical_scale,
                    self.cfg.slope_treshold,
                )
                self.add_trimesh_to_sim(plane_trimesh, origin)
        else:
            plane_size_x = self.heightfield_raw.shape[0] * self.cfg.horizontal_scale
            plane_size_y = self.heightfield_raw.shape[1] * self.cfg.horizontal_scale
            plane_box_size = np.array([plane_size_x, plane_size_y, 0.02])
            plane_trimesh = trimesh.box_trimesh(plane_box_size, plane_box_size / 2)
            self.add_trimesh_to_sim(plane_trimesh, np.zeros(3))

    ##### Helper functions to compute observations that are only available in simulation #####
    def get_difficulty(self, env_row_idx, env_col_idx):
        difficulty = env_row_idx / (self.cfg.num_rows - 1) if self.cfg.curriculum else None
        virtual_terrain = self.track_kwargs["virtual_terrain"]
        return difficulty, virtual_terrain
