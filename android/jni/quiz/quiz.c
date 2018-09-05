#include "cl/cl.h"

#ifdef __cplusplus
extern "C" {
#endif
#include <stdlib.h> 
#include "qlog.h"
#include "quiz.h"
#include <assert.h>
#include <errno.h>
#include <string.h>

#define QUIZ_ERROR 1
#define QUIZ_GLOBAL "quiz"


static int S_push_error(lua_State* L, int quiz_error, int sys_error) {
    char buff[1024];
    if ( 0 == quiz_error ) return 0;
    sprintf(buff,"quiz error %d. %d",quiz_error,sys_error);
    lua_pushstring(L, buff);

    return quiz_error;
}

static int S_push_monster_info(lua_State *L,MonsterInfo *mi) {
    lua_newtable(L);
    lua_pushnumber(L, mi->x);
    lua_rawseti(L, -2, 1);
    lua_pushnumber(L, mi->y);
    lua_rawseti(L, -2, 2);
    lua_pushnumber(L, mi->attr);
    lua_rawseti(L, -2, 3);
    return 1;
    
}
static int S_close_monstersInfo(MonstersInfo *msi) {
    ALOGV("S_close_monstersInfo");
    if (!msi) {
	goto _clear;
    }
    if (msi->mi) {
        free(msi->mi);
    }
_clear:
    free(msi);
    ALOGV("S_close_monstersInfo all clean");
    return 1;

}

static int S_detectMonsterAttributes(lua_State* L) {
	MonstersInfo* msi= NULL;
	const char* path= luaL_checkstring(L, 1);
	msi = cl_recognize(path);
	if (!msi){
		//assert(QUIZ_ERROR);
		S_push_error(L,QUIZ_ERROR,errno);
		lua_pushnil(L);
		lua_insert(L,-2);
		return 2;

	}
	int size=msi->mi_size;
	lua_newtable(L);
	for(int i=0;i<size;i++) {
        	S_push_monster_info(L,&msi->mi[i]);
        	lua_rawseti(L, -2, i+1);

    	}
    	S_close_monstersInfo(msi);

	return 1;
		
}

static int S_tryDetectMonsterAttributes(lua_State* L) {
	ALOGV("S_tryDetectMonsterAttributes");
	MonstersInfo* msi=NULL;
	const char* path= luaL_checkstring(L, 1);
	ALOGV("S_tryDetectMonsterAttributes('%s')",path);
	int i=0,s=0;
	lua_newtable(L);
	ALOGV("cl_recognize");
	msi = cl_recognize(path);
	if (!msi) {
		ALOGV("cl_recognize -> NULL");
		goto bail;
	}
	s = msi->mi_size;
	ALOGV("cl_recognize returns %d results",s);
	for(;i<s;i++) {
		S_push_monster_info(L,&msi->mi[i]);
		lua_rawseti(L, -2, i+1);
	}
	S_close_monstersInfo(msi);
bail:
	return 1;
}
// returns 1 if lua contains our quiz. 0 else
static int checkQuizExists(lua_State *L) {
	lua_getglobal(L, QUIZ_GLOBAL);
	int rv= !lua_isnil(L,-1);
	lua_pop(L,1);
	return rv;
}


static const struct luaL_reg quizlib[] = {
	{"trydetect",S_tryDetectMonsterAttributes},
	{"detect",S_detectMonsterAttributes},
        {NULL,NULL}
};

int luaopen_quiz(lua_State *L) {
	if (!checkQuizExists(L)) {
		ALOGV("initialize cl");
		if (cl_init()) {
			ALOGV("register quiz in lua _G");
			luaL_register(L,QUIZ_GLOBAL,quizlib);
			lua_pop(L,1);// pop table off from stack
		} else {
			ALOGE("can't init cl");
		}
	} else {
		ALOGV("quiz already registered");
	}
        return 1;
}


#ifdef __cplusplus
}
#endif
