

[external]
KcallBC
A
	full_text4
2
0%7 = tail call i64 @_Z13get_global_idj(i32 0) #4
4truncB+
)
	full_text

%8 = trunc i64 %7 to i32
"i64B

	full_text


i64 %7
JcallBB
@
	full_text3
1
/%9 = tail call i64 @_Z12get_local_idj(i32 0) #4
5truncB,
*
	full_text

%10 = trunc i64 %9 to i32
"i64B

	full_text


i64 %9
3icmpB+
)
	full_text

%11 = icmp eq i32 %10, 0
#i32B

	full_text
	
i32 %10
8brB2
0
	full_text#
!
br i1 %11, label %12, label %15
!i1B

	full_text


i1 %11
?bitcast8B2
0
	full_text#
!
%13 = bitcast float* %2 to i32*
Hload8B>
<
	full_text/
-
+%14 = load i32, i32* %13, align 4, !tbaa !8
'i32*8B

	full_text


i32* %13
?store8Bw
u
	full_texth
f
dstore i32 %14, i32* bitcast (float* @normalize_weights_kernel.sumWeights to i32*), align 4, !tbaa !8
%i328B

	full_text
	
i32 %14
'br8B

	full_text

br label %15
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
6icmp8B,
*
	full_text

%16 = icmp slt i32 %8, %1
$i328B

	full_text


i32 %8
:br8B2
0
	full_text#
!
br i1 %16, label %17, label %24
#i18B

	full_text


i1 %16
0shl8B'
%
	full_text

%18 = shl i64 %7, 32
$i648B

	full_text


i64 %7
9ashr8B/
-
	full_text 

%19 = ashr exact i64 %18, 32
%i648B

	full_text
	
i64 %18
\getelementptr8BI
G
	full_text:
8
6%20 = getelementptr inbounds float, float* %0, i64 %19
%i648B

	full_text
	
i64 %19
Lload8BB
@
	full_text3
1
/%21 = load float, float* %20, align 4, !tbaa !8
+float*8B

	full_text


float* %20
mload8Bc
a
	full_textT
R
P%22 = load float, float* @normalize_weights_kernel.sumWeights, align 4, !tbaa !8
Cfdiv8B9
7
	full_text*
(
&%23 = fdiv float %21, %22, !fpmath !12
)float8B

	full_text

	float %21
)float8B

	full_text

	float %22
Lstore8BA
?
	full_text2
0
.store float %23, float* %20, align 4, !tbaa !8
)float8B

	full_text

	float %23
+float*8B

	full_text


float* %20
'br8B

	full_text

br label %24
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 2) #5
4icmp8B*
(
	full_text

%25 = icmp eq i32 %8, 0
$i328B

	full_text


i32 %8
:br8B2
0
	full_text#
!
br i1 %25, label %26, label %31
#i18B

	full_text


i1 %25
Ucall8BK
I
	full_text<
:
8tail call void @cdfCalc(float* %3, float* %0, i32 %1) #6
;sitofp8B/
-
	full_text 

%27 = sitofp i32 %1 to float
Lfdiv8BB
@
	full_text3
1
/%28 = fdiv float 1.000000e+00, %27, !fpmath !12
)float8B

	full_text

	float %27
Ncall8BD
B
	full_text5
3
1%29 = tail call float @d_randu(i32* %5, i32 0) #6
6fmul8B,
*
	full_text

%30 = fmul float %28, %29
)float8B

	full_text

	float %28
)float8B

	full_text

	float %29
Kstore8B@
>
	full_text1
/
-store float %30, float* %4, align 4, !tbaa !8
)float8B

	full_text

	float %30
'br8B

	full_text

br label %31
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 2) #5
:br8B2
0
	full_text#
!
br i1 %11, label %32, label %35
#i18B

	full_text


i1 %11
?bitcast8B2
0
	full_text#
!
%33 = bitcast float* %4 to i32*
Hload8B>
<
	full_text/
-
+%34 = load i32, i32* %33, align 4, !tbaa !8
'i32*8B

	full_text


i32* %33
zstore8Bo
m
	full_text`
^
\store i32 %34, i32* bitcast (float* @normalize_weights_kernel.u1 to i32*), align 4, !tbaa !8
%i328B

	full_text
	
i32 %34
'br8B

	full_text

br label %35
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
:br8B2
0
	full_text#
!
br i1 %16, label %36, label %45
#i18B

	full_text


i1 %16
eload8	B[
Y
	full_textL
J
H%37 = load float, float* @normalize_weights_kernel.u1, align 4, !tbaa !8
;sitofp8	B/
-
	full_text 

%38 = sitofp i32 %8 to float
$i328	B

	full_text


i32 %8
;sitofp8	B/
-
	full_text 

%39 = sitofp i32 %1 to float
Cfdiv8	B9
7
	full_text*
(
&%40 = fdiv float %38, %39, !fpmath !12
)float8	B

	full_text

	float %38
)float8	B

	full_text

	float %39
6fadd8	B,
*
	full_text

%41 = fadd float %40, %37
)float8	B

	full_text

	float %40
)float8	B

	full_text

	float %37
0shl8	B'
%
	full_text

%42 = shl i64 %7, 32
$i648	B

	full_text


i64 %7
9ashr8	B/
-
	full_text 

%43 = ashr exact i64 %42, 32
%i648	B

	full_text
	
i64 %42
\getelementptr8	BI
G
	full_text:
8
6%44 = getelementptr inbounds float, float* %4, i64 %43
%i648	B

	full_text
	
i64 %43
Lstore8	BA
?
	full_text2
0
.store float %41, float* %44, align 4, !tbaa !8
)float8	B

	full_text

	float %41
+float*8	B

	full_text


float* %44
'br8	B

	full_text

br label %45
$ret8
B

	full_text


ret void
&i32*8B

	full_text
	
i32* %5
*float*8B

	full_text

	float* %2
*float*8B

	full_text

	float* %0
*float*8B

	full_text

	float* %4
$i328B

	full_text


i32 %1
*float*8B

	full_text

	float* %3
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
-; undefined function B

	full_text

 
2float8B%
#
	full_text

float 1.000000e+00
#i328B

	full_text	

i32 1
yfloat*8Bk
i
	full_text\
Z
X@normalize_weights_kernel.sumWeights = internal unnamed_addr global float undef, align 4
Yi32*8BM
K
	full_text>
<
:i32* bitcast (float* @normalize_weights_kernel.u1 to i32*)
$i648B

	full_text


i64 32
qfloat*8Bc
a
	full_textT
R
P@normalize_weights_kernel.u1 = internal unnamed_addr global float undef, align 4
ai32*8BU
S
	full_textF
D
Bi32* bitcast (float* @normalize_weights_kernel.sumWeights to i32*)
#i328B

	full_text	

i32 2
#i328B

	full_text	

i32 0        	
 	                    !  "# "$ "" %& '( '' )* )+ ,, -. -- // 01 02 00 34 33 56 78 79 :; :: <= << >? @A @B CD CC EE FG FH FF IJ IK II LM LL NO NN PQ PP RS RT RR UW /X Y Y +Z 3Z 9Z P[ [ +[ ,[ E\ +    
           ! # $ (' *, .- 1/ 20 4 89 ;: = A DC GE HF JB K ML ON QI SP T	 	    &% &) +) 65 67 97 ?> ?@ B@ VU V aa `` V ^^ ]] __ ]] + `` +6 __ 6 __  ^^ ? __ ?& __ &/ aa /b -c c ?d e <f f f Lf Ng Bh i &i 6j j j j 'j /"
normalize_weights_kernel"
_Z13get_global_idj"
_Z12get_local_idj"
_Z7barrierj"	
cdfCalc"	
d_randu*?
6rodinia-3.1-particlefilter-normalize_weights_kernel.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282?

transfer_bytes
???<

wgsize_log1p
@?A

wgsize
?

devmap_label
 
 
transfer_bytes_log1p
@?A