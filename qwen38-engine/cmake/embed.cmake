# Turns a file into a C++ byte array: cmake -DINPUT=file -DOUTPUT=file.cpp -DSYMBOL=kName -P embed.cmake
file(READ "${INPUT}" hex HEX)
string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," arr "${hex}")
file(WRITE "${OUTPUT}" "#include <cstddef>\nextern const unsigned char ${SYMBOL}[] = {${arr}0x00};\nextern const size_t ${SYMBOL}Size = sizeof(${SYMBOL}) - 1;\n")
