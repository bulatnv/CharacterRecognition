# Gaze tracking

# Building and Running

Now let us build and run the complete application and see how it runs all three analysis models.

## Build

1. Open up a terminal (such as xterm) or use an existing terminal to get to a command shell prompt.

2. Change to the directory containing Tutorial Step 4:

```bash
cd /tutorials/inference-tutorials-generic/face_detection_tutorial/Gaze-Detection
```


3. The first step is to configure the build environment for the Intel® Distribution of OpenVINO™ toolkit by running the "setupvars.sh" script.

```bash
source  /opt/intel/openvino/bin/setupvars.sh
```


4. Now we need to create a directory to build the tutorial in and change to it.

```bash
mkdir build
cd build
```


5. The last thing we need to do before compiling is to configure the build settings and build the executable.  We do this by running CMake to set the build target and file locations.  Then, we run Make to build the executable.

```bash
cmake -DCMAKE_BUILD_TYPE=Release ../
make -j4
```


## Run

1. Before running, be sure to source the helper script that will make it easier to use environment variables instead of long names to the models:

```bash
source ../../scripts/setupenv.sh
```


2. You now have the executable file to run ./intel64/Release/gaze_detection.  In order to load the head pose detection model, the "-m_hp" flag needs to be added  followed by the full path to the model.  This model always works with Intel RealSense D415 camera. First, let's run face detection.

```bash
./intel64/Release/gaze_detection -m $mFDA32
```


3. You now have the executable file to run ./intel64/Release/gaze_detection.  In order to load the head pose detection model, the "-m_hp" flag needs to be added  followed by the full path to the model.  This model always works with Intel RealSense D415 camera. And let us also see how easy it is to have the application run a different face detection model by loading the face-detection-retail-0004 IR files by just changing the -m parameter from $mFDA32 to $mFDR32.

```bash
./intel64/Release/gaze_detection -m $mFDR32
```

4. Now let's run with gaze tracking. An OpenCV based gaze tracker based on a simple image gradient-based eye center algorithm by Fabian Timm. Original code for gaze tracking https://github.com/trishume/eyeLike.

```bash
./intel64/Release/gaze_detection -m $mFDA32 -m_hp $mHP32
```


4. You will see rectangles and the head pose axes that follow the faces around the image (if the faces move), accompanied by age and gender results for the faces, and the timing statistics for processing each frame of the video.  

5. [Optional]Finally, if a USB camera has been setup, we can use the application to view live video from the connected USB camera.  The camera is the default source, so we do this by running the application without using any parameters.

```bash
./intel64/Release/face_detection_tutorial -m $mFDA32 -m_ag $mAG32 -m_hp $mHP32 -i cam
```

6. [Optional]Again, you will see colored rectangles drawn around any faces that appear in the images along with the results for age, gender, the axes representing the head poses, and the various render statistics.

# Picking the Right Models for the Right Devices

Throughout this tutorial, we just had the application load the models onto the default CPU device.  Here we will also explore using other devices the GPU and Myriad, if available and setup,.  That brings up several questions that we should discuss to get a more complete idea of how to make the best use of our models and how to optimize the applications using the devices available.

## What Determines the Device a Model Uses?

One of the main factors is going to be the floating point precision that the model requires.  We discuss that below, to answer the question "Are there models that cannot be loaded onto certain devices?"

Another major factor is speed.  Depending on how the model is structured, compiled and optimized, it may lend itself to running faster on a certain device.  Sometimes, you may know that.  Other times, you may have to test the model on different devices to determine where it runs best.

The other major factor in determining where to load a model is parallel processing or load balancing required to meet an application’s performance requirements.  

Once you have made those decisions, you can use the command line arguments to have the application assign the models to the particular device you want them to run on to test and verify.

## How Do I Choose the Specific Device to Run a Model?

In the application, we use command line parameters to specify which device to use for the models we load.  These are "-d", “-d_ag” and “-d_hp”, and they are used for the face detection model, age and gender estimation model, and head pose estimation model, respectively.  The devices that can be specified are “CPU”, “GPU” and “MYRIAD” each of which must be installed and setup before using.

## Are There Models That Cannot be Loaded onto Specific Devices?

Yes.  The main restriction is the precision of the model must be supported by the device.  As we discussed in the Key Concepts section, certain devices can only run models that have the matching floating point precision.  For example, the CPU can only run models that use FP32 precision.  This is because the hardware execution units of the CPU are designed and optimized for FP32 operations.  Similarly, the Myriad can only load models that use FP16 precision.  While the GPU is designed to be more flexible to run both FP16 and FP32 models, though it runs FP16 models faster than FP32 models.

## Are Some Devices Faster Than Others?

The easy answer is "yes."  The more complex answer is that it can be more complex than just “which device is fastest / has the fastest clock speed / and the most cores?”  Some devices are better at certain functions than other devices because of hardware optimizations or internal structures that fit the work being done within a particular model.  As noted previously, devices that can work with models running FP16 can run faster just because they are moving around as little as half the data of when running FP32.

## Are Some Devices Better for Certain Types of Models Than Other Devices?

Again, the easy answer is "yes."  The truth is that it can be difficult to know what model will run best on what device without actually loading the model on a device and seeing how it performs.  This is one of the most powerful features of the Inference Engine and the Intel® Distribution of OpenVINO™ toolkit.  It is very easy to write applications that allow you to get up and running quickly to test many combinations of models and devices, without requiring significant code changes or even recompiling.  Our face detection application can do exactly that.  So let us see what we can learn about how these models work on different devices by running through the options.

### Command Line and All the Arguments

Before we can get started, let us go over the command line parameters again.  We specify the model we want to load by using the "-m*" arguments, which device to load using the “-d*” arguments, and batch size using the “-n*” arguments.  The table below summarizes the arguments for all three models.

<table>
  <tr>
    <td>Model</td>
    <td>Model Argument</td>
    <td>Device Argument</td>
    <td>Batch Size Argument</td>
  </tr>
  <tr>
    <td>Face detection</td>
    <td>-m</td>
    <td>-d</td>
    <td>(none, always set to 1)</td>
  </tr>
  <tr>
    <td>Age and gender</td>
    <td>-m_ag</td>
    <td>-d_ag</td>
    <td>-n_ag</td>
  </tr>
  <tr>
    <td>Head pose</td>
    <td>-m_hp</td>
    <td>-d_hp</td>
    <td>-n_hp</td>
  </tr>
</table>


As we mentioned in the Key Concepts section, the batch size is the number of input data that the models will work on.  For the face detection model, the batch size is fixed to 1.  Even when processing input from a video or a camera, it will only processes a single image/frame at a time.  Depending on the content of the image data, it can return any number of faces.  The application lets us set the batch size on the other models dynamically and the default batch size is 1 for the age and gender and head pose models.  This will work for the Myriad which has a maximum batch size of 1.  Since we are not expecting many results in the test video provided, to simplify things and keep batch size from affecting performance results (something covered in the Car Detection Tutorial), we will use the default batch size of 1 for all models.

Let us look at a sample command line that uses all the parameters so that we can see what it looks like.  For this example, we are running the application from the "step_4/build" directory.

```bash
./intel64/Release/face_detection_tutorial -m $mFDA32 -d GPU -m_ag $mAG16 -d_ag MYRIAD -m_hp $mHP16 -d_hp GPU -i ../../data/head-pose-face-detection-female-and-male.mp4
```


From this command line, we see that the application will load the FP32 face detection model onto the GPU, the FP16 age and gender model on the Myriad, using a batch size of 1, and the FP16 head pose model onto the GPU, with a batch size of 16.  We also specify "-i ../../data/head-pose-face-detection-female-and-male.mp4" so that we have a “known” data set to do our performance tests with.  This MP4 video file used from the Intel® Distribution of OpenVINO™ toolkit samples is a hand-drawn face with a moving camera.  

You can see that it is easy to change the model precision to match the device you want to run it on by changing the model to use the FP16 or FP32 using "16" and “32” built into the names of the variables..  It is easy to make up several test cases to see how the application and each of the inference model, perform.  Just remember that all models run on the CPU must be FP32, and all models run on the Myriad must be FP16.  Models run on the Myriad must also have their batch size set to 1.  Models run on the GPU can be either FP16 or FP32.

### What Kind of Performance Should I See?

That depends on many things, from the specific devices themselves, to the combination of models and devices that you specify, to the other applications running on the target while you collect and process images.  For this tutorial, an Intel® i7-7700 CPU with GPU and USB Intel® Movidius™ Neural Compute Stick limited to USB 2.0 were used.  Results will vary when using other devices, however the general trends should be the same.  That said, let us take a look at some of the performance counts we observed.  

The performance reported in milliseconds and using the "wallclock*" and “totalFramse” variables in the code that time the main loop.  When the application exits, it reports the wallclock time and average time and FPS of main loop for the input image source used.

The following sections go through the command combinations for different devices depending upon which devices are available.

**Note**: It can take a lot of time to run all the commands so the exercise of running and verifying is left to the user.  

#### CPU

**Note**: In order to run this section only the CPU is required.

Command combinations run:

```Bash
# Command line #1
./intel64/Release/face_detection_tutorial -m $mFDA32 -d CPU -m_ag $mAG32 -d_ag CPU -m_hp $mHP32 -d_hp CPU -i ../../data/head-pose-face-detection-female-and-male.mp4
```


When running just the CPU, all models must be assigned to the CPU which gives only one combination to run.  

#### CPU and GPU

**Note**: In order to run this section, the GPU is required to be present and correctly configured.

Command combinations run:

```Bash
# Command line #1
./intel64/Release/face_detection_tutorial -m $mFDA32 -d GPU -m_ag $mAG32 -d_ag CPU -m_hp $mHP32 -d_hp CPU -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #2
./intel64/Release/face_detection_tutorial -m $mFDA32 -d CPU -m_ag $mAG32 -d_ag CPU -m_hp $mHP32 -d_hp GPU -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #3
./intel64/Release/face_detection_tutorial -m $mFDA32 -d CPU -m_ag $mAG32 -d_ag CPU -m_hp $mHP16 -d_hp GPU -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #4
./intel64/Release/face_detection_tutorial -m $mFDA32 -d CPU -m_ag $mAG32 -d_ag GPU -m_hp $mHP32 -d_hp CPU -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #5
./intel64/Release/face_detection_tutorial -m $mFDA16 -d GPU -m_ag $mAG32 -d_ag CPU -m_hp $mHP32 -d_hp CPU -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #6
./intel64/Release/face_detection_tutorial -m $mFDA32 -d CPU -m_ag $mAG16 -d_ag GPU -m_hp $mHP32 -d_hp CPU -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #7
./intel64/Release/face_detection_tutorial -m $mFDA32 -d GPU -m_ag $mAG32 -d_ag GPU -m_hp $mHP32 -d_hp GPU -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #8
./intel64/Release/face_detection_tutorial -m $mFDA32 -d GPU -m_ag $mAG32 -d_ag CPU -m_hp $mHP32 -d_hp GPU -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #9
./intel64/Release/face_detection_tutorial -m $mFDA32 -d GPU -m_ag $mAG32 -d_ag GPU -m_hp $mHP32 -d_hp CPU -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #10
./intel64/Release/face_detection_tutorial -m $mFDA16 -d GPU -m_ag $mAG32 -d_ag CPU -m_hp $mHP16 -d_hp GPU -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #11
./intel64/Release/face_detection_tutorial -m $mFDA16 -d GPU -m_ag $mAG16 -d_ag GPU -m_hp $mHP32 -d_hp CPU -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #12
./intel64/Release/face_detection_tutorial -m $mFDA16 -d GPU -m_ag $mAG16 -d_ag GPU -m_hp $mHP16 -d_hp GPU -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #13
./intel64/Release/face_detection_tutorial -m $mFDA32 -d CPU -m_ag $mAG32 -d_ag GPU -m_hp $mHP32 -d_hp GPU -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #14
./intel64/Release/face_detection_tutorial -m $mFDA32 -d CPU -m_ag $mAG16 -d_ag GPU -m_hp $mHP16 -d_hp GPU -i ../../data/head-pose-face-detection-female-and-male.mp4
```


Performance is measured as the average time for the main loop to process all the input frames.  The average time, and inverse as frames-per-second (fps), with number of frames processed are reported on exit.  The results seen for the configurations listed above should improve starting from the first all the way to the last.  From the end of the list, we see that the fastest results are for the combinations when offloading two models from the CPU and running the age and gender along with head pose models on the GPU using FP16.

#### CPU and Myriad

**Note**: In order to run this section, the Myriad (Intel® Movidius™ Neural Compute Stick) is required to be present and correctly configured.

Command combinations run:

```Bash
# Command line #1
./intel64/Release/face_detection_tutorial -m $mFDA16 -d MYRIAD -m_ag $mAG32 -d_ag CPU -m_hp $mHP32 -d_hp CPU -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #2
./intel64/Release/face_detection_tutorial -m $mFDA16 -d MYRIAD -m_ag $mAG16 -d_ag MYRIAD -m_hp $mHP32 -d_hp CPU -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #3
./intel64/Release/face_detection_tutorial -m $mFDA16 -d MYRIAD -m_ag $mAG32 -d_ag CPU -m_hp $mHP16 -d_hp MYRIAD -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #4
./intel64/Release/face_detection_tutorial -m $mFDA32 -d CPU -m_ag $mAG32 -d_ag CPU -m_hp $mHP16 -d_hp MYRIAD -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #5
./intel64/Release/face_detection_tutorial -m $mFDA32 -d CPU -m_ag $mAG16 -d_ag MYRIAD -m_hp $mHP32 -d_hp CPU -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #6
./intel64/Release/face_detection_tutorial -m $mFDA32 -d CPU -m_ag $mAG16 -d_ag MYRIAD -m_hp $mHP16 -d_hp MYRIAD -i ../../data/head-pose-face-detection-female-and-male.mp4
```


Performance is measured as the average time for the main loop to process all the input frames.  The average time, and inverse as frames-per-second (fps), with number of frames processed are reported on exit.  The results seen for the configurations listed above should improve starting from the first all the way to the last.  From the end of the list, we see that the fastest results are for the combinations when offloading two models from the CPU and running the age and gender along with head pose models on the Myriad.

#### CPU, GPU, and Myriad

**Note**: In order to run this section, the GPU and Myriad (Movidius™ Neural Compute Stick) are required to be present and correctly configured.

Command combinations run:

```Bash
# Command line #1
./intel64/Release/face_detection_tutorial -m $mFDA16 -d MYRIAD -m_ag $mAG32 -d_ag GPU -m_hp $mHP32 -d_hp CPU -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #2
./intel64/Release/face_detection_tutorial -m $mFDA16 -d MYRIAD -m_ag $mAG32 -d_ag CPU -m_hp $mHP32 -d_hp GPU -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #3
./intel64/Release/face_detection_tutorial -m $mFDA16 -d MYRIAD -m_ag $mAG32 -d_ag CPU -m_hp $mHP16 -d_hp GPU -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #4
./intel64/Release/face_detection_tutorial -m $mFDA16 -d MYRIAD -m_ag $mAG16 -d_ag GPU -m_hp $mHP32 -d_hp CPU -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #5
./intel64/Release/face_detection_tutorial -m $mFDA16 -d MYRIAD -m_ag $mAG32 -d_ag GPU -m_hp $mHP32 -d_hp GPU -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #6
./intel64/Release/face_detection_tutorial -m $mFDA16 -d MYRIAD -m_ag $mAG16 -d_ag GPU -m_hp $mHP16 -d_hp GPU -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #7
./intel64/Release/face_detection_tutorial -m $mFDA32 -d GPU -m_ag $mAG16 -d_ag MYRIAD -m_hp $mHP16 -d_hp MYRIAD -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #8
./intel64/Release/face_detection_tutorial -m $mFDA16 -d GPU -m_ag $mAG16 -d_ag MYRIAD -m_hp $mHP16 -d_hp MYRIAD -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #9
./intel64/Release/face_detection_tutorial -m $mFDA32 -d GPU -m_ag $mAG16 -d_ag MYRIAD -m_hp $mHP32 -d_hp CPU -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #10
./intel64/Release/face_detection_tutorial -m $mFDA32 -d GPU -m_ag $mAG32 -d_ag CPU -m_hp $mHP16 -d_hp MYRIAD -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #11
./intel64/Release/face_detection_tutorial -m $mFDA16 -d GPU -m_ag $mAG16 -d_ag MYRIAD -m_hp $mHP32 -d_hp CPU -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #12
./intel64/Release/face_detection_tutorial -m $mFDA16 -d GPU -m_ag $mAG32 -d_ag CPU -m_hp $mHP16 -d_hp MYRIAD -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #13
./intel64/Release/face_detection_tutorial -m $mFDA32 -d CPU -m_ag $mAG16 -d_ag MYRIAD -m_hp $mHP16 -d_hp GPU -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #14
./intel64/Release/face_detection_tutorial -m $mFDA32 -d CPU -m_ag $mAG16 -d_ag MYRIAD -m_hp $mHP32 -d_hp GPU -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #15
./intel64/Release/face_detection_tutorial -m $mFDA32 -d CPU -m_ag $mAG16 -d_ag GPU -m_hp $mHP16 -d_hp MYRIAD -i ../../data/head-pose-face-detection-female-and-male.mp4
# Command line #16
./intel64/Release/face_detection_tutorial -m $mFDA32 -d CPU -m_ag $mAG32 -d_ag GPU -m_hp $mHP16 -d_hp MYRIAD -i ../../data/head-pose-face-detection-female-and-male.mp4
```


Performance is measured as the average time for the main loop to process all the input frames.  The average time, and inverse as frames-per-second (fps), with number of frames processed are reported on exit.  The results seen for the configurations listed above should improve starting from the first all the way to the last.  From the end of the list, we see the fastest results are for the combination  when offloading from the CPU the age and gender model running on GPU and the head pose model running on the MYRIAD.

#### Summary

From the all of the combinations of devices in the above sections, we can see several trends:  

* For the face detection model, performance is fastest on the CPU.  For the age and gender along with head pose models, offloading to GPU or Myriad is faster than the CPU.  

* FP16 performance is almost always faster than FP32 performance on any device.

In general, the best way to maximize performance is to put the most complex model on the fastest device.  Try to divide the work across the devices as much as possible to avoid overloading any one device.  If you do not need FP32 precision, you can speed up your applications by using FP16 models.

Something to note too is that the Myriad is only capable of running two analysis models at a time.  If you try to load a third model, the application will exit, and report a "Device not found" error.

# Conclusion

By adding the head pose estimation model to the application from Tutorial Step 3, you have now seen the final step in assembling the full application.  This again shows the power the Intel® Distribution of OpenVINO™ toolkit brings to applications by quickly being able to add another inference model.  We also discussed how to load the inference models onto different devices to distribute the workload and find the optimal device to get the best performance from the models.

# Navigation

[Face Detection Tutorial](../Readme.md)

[Face Detection Tutorial Step 3](../step_3/Readme.md)
