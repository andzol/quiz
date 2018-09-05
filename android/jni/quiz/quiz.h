#ifndef QUIZ_H
#define QUIZ_H
#ifdef __cplusplus
extern "C" {
#endif

#include "lua.h"
#include "lauxlib.h"
#include "lualib.h"


int getOpenCVInfo(lua_State* L);
int luaopen_quiz(lua_State *L);


#ifdef __cplusplus
}
#endif
#endif
