LOCAL_PATH := $(call my-dir)

 
include $(CLEAR_VARS)

# No static libraries.
LOCAL_STATIC_LIBRARIES := subhook
LOCAL_CFLAGS := -DQUIZDEBUG -Wall -Werror -Wno-unused-function -Wno-logical-op-parentheses
LOCAL_C_INCLUDES +=$(LOCAL_PATH)/lua5.1/ \
	$(HOME)/OpenCV-android-sdk/sdk/native/jni/include/ \
	$(LOCAL_PATH)/../subhook/ \
	$(LOCAL_PATH)/cl/
#LOCAL_NDK_STL_VARIANT := none
LOCAL_SDK_VERSION := current
LOCAL_LDLIBS := -L$(SYSROOT)/usr/lib
LOCAL_LDLIBS += -L$(LOCAL_PATH)/libs -lopencv_info -lopencv_java -lluajava -ldl  -llog -lz
#LOCAL_ARM_MODE := arm
LOCAL_MODULE_TAGS := quiz
LOCAL_MODULE    := quiz
LOCAL_SRC_FILES := inject_lua.c \
	cl/load_classifier.cpp \
	cl/cl.cc \
	cl/feat.cc \
	quiz.c \
	jni_load.cpp

include $(BUILD_SHARED_LIBRARY)

