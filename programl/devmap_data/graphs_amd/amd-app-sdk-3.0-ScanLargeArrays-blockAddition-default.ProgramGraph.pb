

[external]
KcallBC
A
	full_text4
2
0%3 = tail call i64 @_Z13get_global_idj(i32 0) #3
JcallBB
@
	full_text3
1
/%4 = tail call i64 @_Z12get_local_idj(i32 0) #3
4truncB+
)
	full_text

%5 = trunc i64 %4 to i32
"i64B

	full_text


i64 %4
1icmpB)
'
	full_text

%6 = icmp eq i32 %5, 0
"i32B

	full_text


i32 %5
6brB0
.
	full_text!

br i1 %6, label %7, label %14
 i1B

	full_text	

i1 %6
Lcall8BB
@
	full_text3
1
/%8 = tail call i64 @_Z12get_group_idj(i32 0) #3
/shl8B&
$
	full_text

%9 = shl i64 %8, 32
$i648B

	full_text


i64 %8
8ashr8B.
,
	full_text

%10 = ashr exact i64 %9, 32
$i648B

	full_text


i64 %9
\getelementptr8BI
G
	full_text:
8
6%11 = getelementptr inbounds float, float* %0, i64 %10
%i648B

	full_text
	
i64 %10
@bitcast8B3
1
	full_text$
"
 %12 = bitcast float* %11 to i32*
+float*8B

	full_text


float* %11
Hload8B>
<
	full_text/
-
+%13 = load i32, i32* %12, align 4, !tbaa !8
'i32*8B

	full_text


i32* %12
tstore8Bi
g
	full_textZ
X
Vstore i32 %13, i32* bitcast (float* @blockAddition.value.0 to i32*), align 4, !tbaa !8
%i328B

	full_text
	
i32 %13
'br8B

	full_text

br label %14
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
_load8BU
S
	full_textF
D
B%15 = load float, float* @blockAddition.value.0, align 4, !tbaa !8
0shl8B'
%
	full_text

%16 = shl i64 %3, 32
$i648B

	full_text


i64 %3
9ashr8B/
-
	full_text 

%17 = ashr exact i64 %16, 32
%i648B

	full_text
	
i64 %16
\getelementptr8BI
G
	full_text:
8
6%18 = getelementptr inbounds float, float* %1, i64 %17
%i648B

	full_text
	
i64 %17
Lload8BB
@
	full_text3
1
/%19 = load float, float* %18, align 4, !tbaa !8
+float*8B

	full_text


float* %18
6fadd8B,
*
	full_text

%20 = fadd float %15, %19
)float8B

	full_text

	float %15
)float8B

	full_text

	float %19
Lstore8BA
?
	full_text2
0
.store float %20, float* %18, align 4, !tbaa !8
)float8B

	full_text

	float %20
+float*8B

	full_text


float* %18
$ret8B

	full_text


ret void
*float*8B

	full_text

	float* %0
*float*8B

	full_text

	float* %1
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
#i328B

	full_text	

i32 1
kfloat*8B]
[
	full_textN
L
J@blockAddition.value.0 = internal unnamed_addr global float undef, align 4
$i648B

	full_text


i64 32
#i328B

	full_text	

i32 0
Si32*8BG
E
	full_text8
6
4i32* bitcast (float* @blockAddition.value.0 to i32*)       	 
 

                      !" !# !! $% $& $$ '( )    	 
           " #! % & 	   ' ** ,, ++ -- ** 	 ++ 	 --  ,, . / 0 
0 0 0 1 1 1 1 	2 "
blockAddition"
_Z13get_global_idj"
_Z12get_group_idj"
_Z12get_local_idj"
_Z7barrierj*?
 ScanLargeArrays-blockAddition.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282?

transfer_bytes
??

wgsize
?

wgsize_log1p
W?GA

devmap_label
 
 
transfer_bytes_log1p
W?GA