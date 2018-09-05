#ifndef INJECTLUA_H
#define INJECTLUA_H
#ifdef __cplusplus
extern "C" {
#endif

#include "jni.h"

int injectLuaJava(JNIEnv* env);
void freeSubhook();
#ifdef __cplusplus
}
#endif

#endif


