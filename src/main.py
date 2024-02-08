import asyncio

from viam.services.vision import Vision
from viam.module.module import Module
from viam.resource.registry import Registry, ResourceCreatorRegistration
from .motion_detector.motion_detector import MotionDetector


async def main():
    """
    This function creates and starts a new module, after adding all desired
    resource models. Resource creators must be registered to the resource
    registry before the module adds the resource model.
    """
    print("hello1")
    Registry.register_resource_creator(
        Vision.SUBTYPE,
        MotionDetector.MODEL,
        ResourceCreatorRegistration(MotionDetector.new_service, MotionDetector.validate_config))
    print("hello2")
    module = Module.from_args()
    print("hello3")

    module.add_model_from_registry(Vision.SUBTYPE, MotionDetector.MODEL)
    print("hello4") 
    await module.start()

if __name__ == "__main__":
    asyncio.run(main())
