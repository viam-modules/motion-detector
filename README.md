# motion-detector

Viam provides a `motion-detector` model of the [vision service](/services/vision) with which you can detect the occurance and location of movement.

To transform your camera into a motion detecting camera, configure this vision service as a [modular resource](https://docs.viam.com/modular-resources/) on your robot.

## Getting started

Start by [configuring a camera](https://docs.viam.com/components/camera/webcam/) on your robot. Remember the name you give to the camera, it will be important later.

> [!NOTE]
> Before configuring your camera or vision service, you must [create a robot](https://docs.viam.com/manage/fleet/robots/#add-a-new-robot).

> [!NOTE]
> If you run this on a non-Debian-based flavor of Linux, you need to ensure that libGL.so.1 is installed on your system! It's probably already installed, unless you're using a headless setup. Ubuntu is Debian-based, so this note doesn't apply on Ubuntu.

## Configuration

Navigate to the **Config** tab of your robot’s page in [the Viam app](https://app.viam.com/). Click on the **Services** subtab and click **Create service**. Select the `vision` type, then select the `motion-detector` model. Enter a name for your service and click **Create**.

On the new component panel, copy and paste the following attribute template into your base’s **Attributes** box.
```json
{
  "camera_name": "myCam",
  "sensitivity": 0.9,
  "min_box_size": 2000
}
```

Edit the attributes as applicable.

> [!NOTE]
> For more information, see [Configure a Robot](https://docs.viam.com/manage/configuration/).

### Attributes

The following attributes are available for `viam:vision:motion-detector` vision services:

| Name | Type | Inclusion | Description |
| ---- | ---- | --------- | ----------- |
| `camera_name` | string | **Required** | The name of the camera configured on your robot. |
| `cam_name` | string | **Required** | \*\***DEPRECATED**\*\* The name of the camera configured on your robot. |
| `min_box_size` | int | **Optional** | The size (in square pixels) of the smallest bounding box to allow. Relevant for GetDetections/GetDetectionsFromCamera only. You must specify at most one of `min_box_size` and `min_box_percent`.
| `min_box_percent` | int | **Optional** | The fraction of the image (between 0 and 1) that the smallest bounding box must cover. Relevant for GetDetections/GetDetectionsFromCamera only. You must specify at most one of `min_box_size` and `min_box_percent`.
| `max_box_size` | int | **Optional** | The size (in square pixels) of the largest bounding box to allow. Relevant for GetDetections/GetDetectionsFromCamera only. You must specify at most one of `max_box_size` and `max_box_percent`.
| `max_box_percent` | int | **Optional** | The fraction of the image (between 0 and 1) that the largest bounding box can cover. Relevant for GetDetections/GetDetectionsFromCamera only. You must specify at most one of `max_box_size` and `max_box_percent`.
| `sensitivity` | float | **Optional** | A number from 0 - 1. Larger numbers will make the module more sensitive to motion. Default = 0.9 |

> [!WARNING]  
> Either one of `camera_name` or `cam_name` will be accepted, but not both. `camera_name` is preferred.

### Example Configuration

```json
{
  "modules": [
    {
      "type": "registry",
      "name": "viam_motion-detector",
      "module_id": "viam:motion-detector",
      "version": "0.0.3"
    }
  ],
  "services": [
    {
      "name": "myMotionDetectionModule",
      "type": "vision",
      "namespace": "rdk",
      "model": "viam:vision:motion-detector",
      "attributes": {
        "camera_name": "myCam",
        "sensitivity": 0.9,
        "min_box_size": 2000
      }
    }
  ]
}

```

### Example Attributes

You should be able to copy and paste this straight into the setup section, just change the camera name:

```json
{
  "camera_name": "myCam",
  "sensitivity": 0.9,
  "min_box_size": 2000
}
```

### Usage

This module is made for use with the following methods of the [vision service API](https://docs.viam.com/services/vision/#api):
- [`GetClassifications()`](https://docs.viam.com/services/vision/#getclassifications)
- [`GetClassificationsFromCamera()`](https://docs.viam.com/services/vision/#getclassificationsfromcamera)
- [`GetDetections()`](https://docs.viam.com/services/vision/#getdetections)
- [`GetDetectionsFromCamera()`](https://docs.viam.com/services/vision/#getdetectionsfromcamera)


The module behavior differs slightly for classifications and detections.

When returning classifications, the module will always return a single classification with the `class_name` "motion".
The `confidence` of the classification will be a percentage equal to the percentage of the image that moved (more than a threshold determined by the sensitivity attribute).

When returning detections, the module will return a list of detections with bounding boxes that encapsulate the movement.
The `class_name` will be "motion" and the `confidence` will always be 0.5.

## Visualize

Once the `viam:vision:motion-detector` modular service is in use, configure a [transform camera](https://docs.viam.com/components/camera/transform/) to see classifications or detections appear in your robot's field of vision.

## Next Steps

- To test your motion detector, configure a [transform camera](https://docs.viam.com/components/camera/transform/) to see classifications or detections appear in your robot's field of vision from the [**Control** tab](https://docs.viam.com/manage/fleet/robots/#control).
- To write code to use the motion detector output, use one of the [available SDKs](https://docs.viam.com/program/).

## License

Copyright 2021-2023 Viam Inc. <br>
Apache 2.0
