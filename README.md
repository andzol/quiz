# quiz

Pretty simple SW quiz event solver PoC using ankulua core

# Monsters Type detection

a lot of work is done.

currently i have a pretty good model on monster attr type recognition, using RandomForest classifiers. some improvements may be good.
this part ( classifier training) is implmemented using MSVS 2017 Community Edition
there are a lot of code parts here and there. the majority of this - is an ability to create pretrained classified.

most interesting parts in these code are in 
/features_extract

we train classified. then save it. then use saved data to create c-style include header with classifier data(model.h)
and then we do use this model.h in our android code


# Android

(see /android folder)

as for now i do all android dev using x86 android image on oracle virtualbox.

the idea is to:
* create some subroutines to get image from android screen
* detect monsters there and  return something like (x,y,type) for each monster found.

this is done in c/c++ using luajava ability to load arbitrary shared libraries in runtime.
i use this to inject my code into ankulua core and give lua scripts access to quiz solver routines.(see /android/test.lua)
i use "build in" ankulua native libs (luajava, openCV).
i statically link only zlib(because non-zipped model >10Mb ) and c++ libs

as for now

* implemented code injection into ankulua (x86 only)
* implemented lua c-module to recognize monsters from image data, used by ankulua


### TODO

* check wether this can be run using non-root x86 emulators
* test/tune this code after testing more quiz images( i've found a few)
* improve recognition accuracy( i moved some monsters from classifier train set into test set)
* improve monster icons detection on quiz page for various screen resolutions( look for findSquares function in cl.cc )
* maybe i forgot something :)
* make all code looks nicer

Hope this helps 