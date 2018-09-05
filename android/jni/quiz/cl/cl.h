#pragma once

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif


typedef enum MonsterType_t {
	WATER=1,
	FIRE=2,
	DARK=3,
	LIGHT=4,
	WIND=5
} MonsterType;

typedef struct MonsterInfo_t {
	MonsterType attr;
	int x;
	int y;
} MonsterInfo;


typedef struct MonstersInfo_t {
	MonsterInfo* mi;
	int mi_size;
} MonstersInfo;

EXTERNC MonstersInfo* cl_recognize(const char* filename);
EXTERNC int cl_init(); 

#undef EXTERNC
