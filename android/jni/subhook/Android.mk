LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

# No static libraries.
LOCAL_STATIC_LIBRARIES :=
LOCAL_CFLAGS := -Wall -Werror
LOCAL_NDK_STL_VARIANT := none
LOCAL_SDK_VERSION := current
#LOCAL_ARM_MODE := arm
LOCAL_MODULE_TAGS := subhook
LOCAL_MODULE    := subhook
LOCAL_SRC_FILES := \
	subhook.c 
include $(BUILD_STATIC_LIBRARY)
#
