#ifdef __cplusplus
extern "C" {
#endif

#include "qlog.h"
#include "inject_lua.h"
#include "subhook.h"
#include "lua.h"
#include "quiz.h"

static subhook_t lua_type_hook;
static int hook_installed = 0;
typedef int (*lua_type_ptr)(lua_State*,int);


static void stackdump(lua_State* l)
{
    int i;
    int top = lua_gettop(l);

    for (i = 1; i <= top; i++)
    {  //repeat for each level
        int t = lua_type(l, i);
        switch (t) {
        case LUA_TSTRING:  // strings 
            ALOGD("\tstring: '%s'\n", lua_tostring(l, i));
            break;
        case LUA_TBOOLEAN:  // booleans //
            ALOGD("\tboolean %s\n",lua_toboolean(l, i) ? "true" : "false");
            break;
        case LUA_TNUMBER:  // numbers //
            ALOGD("\tnumber: %g\n", lua_tonumber(l, i));
            break;
        default:  // other values 
            ALOGD("\t%s\n", lua_typename(l, t));
            break;
        }
    }
}


int my_lua_type(lua_State* L, int index) {
//	freeSubhook();
	subhook_remove(lua_type_hook);
	luaopen_quiz(L);
	//stackdump(L);
	int retval = lua_type(L,index);
	//luaopen_quiz(L);
	subhook_install(lua_type_hook);
	//ALOGV("my_lua_type called. and returns: %d",retval);
	return retval;
}

void freeSubhook() {
	if (hook_installed) {
		subhook_remove(lua_type_hook);
		subhook_free(lua_type_hook);
		ALOGV("hooks removed and cleaned");
	}
}


//return JNI_TRUE or JNI_FALSE
int injectLuaJava(JNIEnv* env) {

	lua_type_hook=subhook_new((void*)lua_type,(void*)my_lua_type,(subhook_options_t)0);
	if (subhook_install(lua_type_hook)<0) {
		ALOGE("can't install hook on lua_type");
		return JNI_FALSE;
	}
	hook_installed=1;
	return JNI_TRUE;

}
#ifdef __cplusplus
}
#endif

