function _log(...) 

local f=io.open('/sdcard/AnkuLua/quiztest.log','ab+')
if (f) then
  f:write(...)
  f:write("\n")
  f:close()
end
end
function _rtLoadQuiz() 
  if (quiz == nil) then
	  local sl=luajava.bindClass('java.lang.System')
	  if (sl ~=nil) then
	   sl:load('/data/local/tmp/libquiz.so')
	  end
	  type('init_quiz')
  end
  return quiz
end
function _checkSnapshot1(f)
 local s=io.open(f,'r')
 if (s) then
  s:close()
  _log('snapshot found '.. f)
  return 1
 end
 return 0
end

function _checkSnapshot()
 local p='/data/local/tmp/'
 if (_checkSnapshot1(p..'ankulua.raw')==1) then
 	return p ..'ankulua.raw'
 end
 if (_checkSnapshot1(p..'rootankulua.raw')==1) then
	return p .. 'rootankulua.raw'
 end
 return nil
end

_log('try load libquiz.so')

if (_rtLoadQuiz() == nil) then
 _log('error init quiz')
 toast('error. ups')
 scriptExit()
end

_log('libquiz loaded')
_log('taking snapshot')
--ankulua error snapshot() --/data/local/tmp/ankulua.raw update
usePreviousSnap(false)
usePreviousSnap(true)
usePreviousSnap(false)
raw=_checkSnapshot()
if (raw == nil ) then
  _log('no snapshot found')
  scriptExit()
end

_log('trydetect from '..raw)
results=quiz.trydetect(raw)
_log('detect done.check results')
_log(result ~=nil and 'some result' or 'nil value')
for k,v in ipairs(results) do
  _log(k,v ~=nil and 'some result' or 'nil value')
end

_log('all done i think')
