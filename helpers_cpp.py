import cppyy

cpp_code = open("helpers.cpp").read()
cppyy.cppdef(cpp_code)

decode_prob_cpp = cppyy.gbl.decode_prob
encode_prob_cpp = cppyy.gbl.encode_prob