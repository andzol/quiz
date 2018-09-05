#!/bin/bash

_done() {
  adb pull /sdcard/AnkuLua/quiztest.log .
}
trap _done 'EXIT'

adb push test.lua /sdcard/AnkuLua/
adb push $(find ./libs/ -name "*.so") /data/local/tmp/
frida -f com.appautomatic.ankulua.pro2 -U --no-pause
