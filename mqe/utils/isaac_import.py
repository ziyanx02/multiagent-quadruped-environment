try:
    from isaacgym import gymapi, gymutil, gymtorch
except ImportError:
    try:
        from omni.isaac.lab import gymapi, gymutil, gymtorch
    except ImportError as e:
        raise ImportError(
            "Neither Isaac Gym nor Isaac Lab is installed. Please install one of them to use MQE."
        ) from e
