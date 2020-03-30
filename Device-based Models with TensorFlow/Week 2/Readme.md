### Check the google developer courses:  
https://developer.android.com/courses


### Latest Tensorflow lite android library:  
https://bintray.com/google/tensorflow/tensorflow-lite


##### Be shure not to compress the TFLite model files by compiler (gradle):  
```javascript
dependencies{  
  implementation 'org.tensorflow:tensorflow-lite:2.1.0'  
}  
android {  
     apptOptions{  
        noCompress "tflite"  // Model extension  
     }  
}
```
