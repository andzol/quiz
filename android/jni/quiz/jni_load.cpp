/*
 * Copyright (C) 2008 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "qlog.h"
#include "jni.h"
#include "inject_lua.h"

static jobject g_classLoader = NULL;
static jmethodID g_findClassMethod = NULL;


void getCurrentThreadClassLoader(JNIEnv* env) {
	jclass threadCls = env->FindClass("java/lang/Thread");
	jmethodID currentThreadMid = env->GetStaticMethodID(threadCls, "currentThread", "()Ljava/lang/Thread;");
	jobject currentThread = env->CallStaticObjectMethod(threadCls, currentThreadMid);
	jmethodID getCtxClsLoaderMid = env->GetMethodID(threadCls, "getContextClassLoader", "()Ljava/lang/ClassLoader;");
	g_classLoader = env->CallObjectMethod(currentThread, getCtxClsLoaderMid);
	if (g_classLoader == NULL) 
		return;
	jclass clsObj = env->GetObjectClass(g_classLoader);
	if (clsObj == NULL) 
		return;
	g_findClassMethod = env->GetMethodID(clsObj, "findClass", "(Ljava/lang/String;)Ljava/lang/Class;");
}

jclass FindClass(JNIEnv* env, const char* classStr)
{

	jclass cl = (jclass) env->CallObjectMethod(g_classLoader, g_findClassMethod, env->NewStringUTF(classStr));
	// Workaround for bug in sun 1.6 VMs
	return cl;
}


/*
 * Register several native methods for one class.
 */
static int saveGlobalClassLoader(JNIEnv* env)
{

getCurrentThreadClassLoader(env);
if (g_classLoader == NULL) {
 ALOGE("can't find current thread classloader");
 return JNI_FALSE;
}

const char* clazz[]={
 "java/lang/System",
 "org/opencv/core/Core",
 "org/keplerproject/luajava/LuaStateFactory",
};
    for (int i=0;i<sizeof(clazz)/sizeof(clazz[0]);i++) {
     jclass lsf = FindClass(env,clazz[i]);
     if (lsf == NULL) {
    	ALOGE("not found %s",clazz[i]);
	return JNI_FALSE;
     } else {
	ALOGV("found %s",clazz[i]);
     }
    }
    return JNI_TRUE;
}
// ----------------------------------------------------------------------------
/*
 * This is called by the VM when the shared library is first loaded.
 */
 
typedef union {
    JNIEnv* env;
    void* venv;
} UnionJNIEnvToVoid;
jint JNI_OnLoad(JavaVM* vm, void* /*reserved*/)
{
    UnionJNIEnvToVoid uenv;
    uenv.venv = NULL;
    jint result = -1;
    JNIEnv* env = NULL;
    
    ALOGI("JNI_OnLoad");
    if (vm->GetEnv(&uenv.venv, JNI_VERSION_1_4) != JNI_OK) {
        ALOGE("ERROR: GetEnv failed");
        goto bail;
    }
    env = uenv.env;
/*    if (saveGlobalClassLoader(env) != JNI_TRUE) {
      ALOGE("ERROR: can't save global class loader");
      goto bail;
    }
*/
    if (injectLuaJava(env) != JNI_TRUE) {
        ALOGE("ERROR: injectLuaJava failed");
        goto bail;
    }
    
    result = JNI_VERSION_1_4;
    
bail:
    return result;
}

