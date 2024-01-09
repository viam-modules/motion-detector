# motion-detector

Viam provides a `motion-detector` model of the [vision service](/services/vision) with which you can detect the occarance and location of movement.

To transform your camera into a motion detecting camera, configure this vision service as a [modular resource](https://docs.viam.com/modular-resources/) on your robot.

## Getting started

Start by [configuring a camera](https://docs.viam.com/components/camera/webcam/) on your robot. Remember the name you give to the camera, it will be important later.

> [!NOTE]  
> Before configuring your camera or vision service, you must [create a robot](https://docs.viam.com/manage/fleet/robots/#add-a-new-robot).

## Configuration

Navigate to the **Config** tab of your robot’s page in [the Viam app](https://app.viam.com/). Click on the **Services** subtab and click **Create service**. Select the `vision` type, then select the `motion-detector` model. Enter a name for your service and click **Create**.

On the new component panel, copy and paste the following attribute template into your base’s **Attributes** box. 
```json
{
  "cam_name": "myCam",
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
| `cam_name` | string | **Required** | The name of the camera configured on your robot. |
| `min_box_size` | int | **Required** | The size (in square pixels) of the smallest bounding box to allow. Relevant for GetDetections/GetDetectionsFromCamera only.
| `sensitivity` | float | **Optional** | A number from 0 - 1. Larger numbers will make the module more sensitive to motion. Default = 0.9 |

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
        "cam_name": "myCam",
        "sensitivity": 0.9,
        "min_box_size": 2000
      }
    }
  ]
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
