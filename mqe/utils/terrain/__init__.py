import importlib

terrain_registry = dict(
    Terrain= "mqe.utils.terrain.terrain:Terrain",
    BarrierTrack= "mqe.utils.terrain.barrier_track:BarrierTrack",
    TerrainPerlin= "mqe.utils.terrain.perlin:TerrainPerlin",
)

def get_terrain_cls(terrain_cls):
    entry_point = terrain_registry[terrain_cls]
    module, class_name = entry_point.rsplit(":", 1)
    module = importlib.import_module(module)
    return getattr(module, class_name)
