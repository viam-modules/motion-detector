import asyncio

from viam.services.vision import Vision
from viam.module.module import Module
from viam.resource.registry import Registry, ResourceCreatorRegistration
from src.motion_detector import MotionDetector


async def main():
    """
    This function creates and starts a new module, after adding all desired
    resource models. Resource creators must be registered to the resource
    registry before the module adds the resource model.
    """
    Registry.register_resource_creator(
        Vision.API,
        MotionDetector.MODEL,
        ResourceCreatorRegistration(
            MotionDetector.new_service, MotionDetector.validate_config
        ),
    )
    module = Module.from_args()

    module.add_model_from_registry(Vision.API, MotionDetector.MODEL)
    await module.start()


if __name__ == "__main__":
    asyncio.run(main())
