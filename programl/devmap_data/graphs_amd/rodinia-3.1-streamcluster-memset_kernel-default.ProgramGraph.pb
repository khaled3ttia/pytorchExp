

[external]
KcallBC
A
	full_text4
2
0%4 = tail call i64 @_Z13get_global_idj(i32 0) #2
3truncB*
(
	full_text

%5 = trunc i16 %1 to i8
-shlB&
$
	full_text

%6 = shl i64 %4, 32
"i64B

	full_text


i64 %4
5ashrB-
+
	full_text

%7 = ashr exact i64 %6, 32
"i64B

	full_text


i64 %6
RgetelementptrBA
?
	full_text2
0
.%8 = getelementptr inbounds i8, i8* %0, i64 %7
"i64B

	full_text


i64 %7
BstoreB9
7
	full_text*
(
&store i8 %5, i8* %8, align 1, !tbaa !8
 i8B

	full_text	

i8 %5
"i8*B

	full_text


i8* %8
"retB

	full_text


ret void
$i8*8B

	full_text


i8* %0
$i168B

	full_text


i16 %1
-; undefined function B

	full_text

 
$i648B

	full_text


i64 32
#i328B

	full_text	

i32 0        	
 	 		       
        "
memset_kernel"
_Z13get_global_idj*?
*rodinia-3.1-streamcluster-memset_kernel.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282?
 
transfer_bytes_log1p
n??A

wgsize_log1p
n??A

devmap_label
 

wgsize
?

transfer_bytes
???&