from src.motion_detector import MotionDetector
from tests.fakecam import FakeCamera
from PIL import Image
from unittest.mock import MagicMock, patch
from viam.components.camera import Camera
from viam.proto.app.robot import ComponentConfig
from google.protobuf.struct_pb2 import Struct
from viam.services.vision import CaptureAllResult, Classification, Detection
from typing import List, Mapping, Any

from parameterized import parameterized
import pytest
import cv2
import numpy as np


def make_component_config(dictionary: Mapping[str, Any]) -> ComponentConfig:
        struct = Struct()
        struct.update(dictionary=dictionary)
        return ComponentConfig(attributes=struct)


def getMD():
    md = MotionDetector("test")
    md.sensitivity = 0.9
    md.min_box_size    = 1000
    md.min_box_percent = 0
    md.max_box_size    = 0
    md.max_box_percent = 0
    md.cam_name = "test"
    md.camera = FakeCamera("test")
    return md


class TestConfigValidation:
    def test_empty(self):
        md = getMD()
        empty_config = make_component_config({})
        with pytest.raises(ValueError, match="Source camera must be provided as 'cam_name'"):
            response = md.validate_config(config=empty_config)


    # For each way to specify a valid min/max size, have a test that checks it's valid.
    @parameterized.expand((
        ("all defaults",                     {}),
        ("min in pixels",                    {"min_box_size": 3}),
        ("min in percentage",                {"min_box_percent": 0.1}),
        ("max in pixels",                    {"max_box_size": 300}),
        ("max in percentage",                {"max_box_percent": 0.9}),
        ("min and max in pixels",            {"min_box_size": 3, "max_box_size": 300}),
        ("min in pixels, max in percentage", {"min_box_size": 3, "max_box_percent": 0.9}),
        ("min and max in percentage",        {"min_box_percent": 0.1, "max_box_percent": 0.9}),
        ("min in percentage, max in pixels", {"min_box_percent": 0.1, "max_box_size": 300}),
    ))
    def test_valid(self, unused_test_name, extra_config_values):
        md = getMD()
        raw_config = {"cam_name": "test"}
        raw_config.update(extra_config_values)
        config = make_component_config(raw_config)
        response = md.validate_config(config=config)
        assert response == ["test"]


    # For each type of invalid config, test that the expected error is raised.
    @parameterized.expand((
        ("Minimum bounding box size should be a non-negative integer", {"min_box_size": -1}),
        ("Minimum bounding box percent should be between 0.0 and 1.0", {"min_box_percent": -0.1}),
        ("Minimum bounding box percent should be between 0.0 and 1.0", {"min_box_percent": 1.1}),
        ("Maximum bounding box size should be a non-negative integer", {"max_box_size": -1}),
        ("Maximum bounding box percent should be between 0.0 and 1.0", {"max_box_percent": -0.1}),
        ("Maximum bounding box percent should be between 0.0 and 1.0", {"max_box_percent": 1.1}),
        ("Cannot specify the minimum box in both pixels and percentages",
            {"min_box_size": 3, "min_box_percent": 0.1}),
        ("Cannot specify the maximum box in both pixels and percentages",
            {"max_box_size": 300, "max_box_percent": 0.9}),
    ))
    def test_invalid(self, error_message, extra_config_values):
        md = getMD()
        raw_config = {"cam_name": "test"}
        raw_config.update(extra_config_values)
        config = make_component_config(raw_config)
        with pytest.raises(ValueError, match=error_message):
            response = md.validate_config(config=config)


class TestMotionDetector:
    @staticmethod
    async def get_output(md):
        out = await md.capture_all_from_camera("test",return_image=True,
                                                return_classifications=True,
                                                return_detections=True,
                                                return_object_point_clouds=True)
        assert isinstance(out, CaptureAllResult)
        assert out.image is not None
        assert out.classifications is not None
        assert len(out.classifications) == 1
        assert out.classifications[0]["class_name"] == "motion"
        return out


    def test_classifications(self):
        img1 = Image.open("tests/img1.jpg")
        img2 = Image.open("tests/img2.jpg")
        gray1 = cv2.cvtColor(np.array(img1), cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(np.array(img2), cv2.COLOR_BGR2GRAY)

        md = getMD()
        classifications = md.classification_from_gray_imgs(gray1, gray2)
        assert len(classifications) == 1
        assert classifications[0]["class_name"] == "motion"


    def test_detections(self):
        img1 = Image.open("tests/img1.jpg")
        img2 = Image.open("tests/img2.jpg")
        gray1 = cv2.cvtColor(np.array(img1), cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(np.array(img2), cv2.COLOR_BGR2GRAY)

        md = getMD()
        detections = md.detections_from_gray_imgs(gray1, gray2)
        assert len(detections) > 0
        assert detections[0]["class_name"] == "motion"


    @pytest.mark.asyncio
    async def test_properties(self):
        md = getMD()
        props = await md.get_properties()
        assert props.classifications_supported == True
        assert props.detections_supported == True
        assert props.object_point_clouds_supported == False


    @pytest.mark.asyncio
    async def test_captureall(self):
        md = getMD()
        out = await self.get_output(md)
        assert out.detections is not None
        assert out.detections[0]["class_name"] == "motion"
        assert out.objects is None


    @pytest.mark.asyncio
    async def test_captureall_not_too_large(self):
        md = getMD()
        md.max_box_size = 1000000000
        out = await self.get_output(md)
        assert out.detections is not None
        assert out.detections[0]["class_name"] == "motion"
        assert out.objects is None


    @pytest.mark.asyncio
    async def test_captureall_too_small(self):
        md = getMD()
        md.min_box_size = 1000000000
        out = await self.get_output(md)
        assert out.detections == []


    @pytest.mark.asyncio
    async def test_captureall_too_large(self):
        md = getMD()
        md.max_box_size = 5
        out = await self.get_output(md)
        assert out.detections == []
