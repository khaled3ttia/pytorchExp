

[external]
JcallBB
@
	full_text3
1
/%7 = tail call i64 @_Z12get_group_idj(i32 0) #4
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
3icmpB+
)
	full_text

%9 = icmp ult i32 %8, %3
"i32B

	full_text


i32 %8
7brB1
/
	full_text"
 
br i1 %9, label %10, label %16
 i1B

	full_text	

i1 %9
Mcall8BC
A
	full_text4
2
0%11 = tail call i64 @_Z12get_local_idj(i32 0) #4
8trunc8B-
+
	full_text

%12 = trunc i64 %11 to i32
%i648B

	full_text
	
i64 %11
7icmp8B-
+
	full_text

%13 = icmp ult i32 %12, %2
%i328B

	full_text
	
i32 %12
\getelementptr8BI
G
	full_text:
8
6%14 = getelementptr inbounds float, float* %5, i64 %11
%i648B

	full_text
	
i64 %11
5icmp8B+
)
	full_text

%15 = icmp eq i64 %11, 0
%i648B

	full_text
	
i64 %11
'br8B

	full_text

br label %17
$ret8B

	full_text


ret void
Cphi8B:
8
	full_text+
)
'%18 = phi i32 [ %8, %10 ], [ %63, %59 ]
$i328B

	full_text


i32 %8
%i328B

	full_text
	
i32 %63
Cphi8B:
8
	full_text+
)
'%19 = phi i64 [ %7, %10 ], [ %62, %59 ]
$i648B

	full_text


i64 %7
%i648B

	full_text
	
i64 %62
1mul8B(
&
	full_text

%20 = mul i32 %18, %2
%i328B

	full_text
	
i32 %18
6zext8B,
*
	full_text

%21 = zext i32 %20 to i64
%i328B

	full_text
	
i32 %20
\getelementptr8BI
G
	full_text:
8
6%22 = getelementptr inbounds float, float* %0, i64 %21
%i648B

	full_text
	
i64 %21
:br8B2
0
	full_text#
!
br i1 %13, label %23, label %25
#i18B

	full_text


i1 %13
Ocall8BE
C
	full_text6
4
2%24 = tail call i64 @_Z14get_local_sizej(i32 0) #4
'br8B

	full_text

br label %29
Ophi8BF
D
	full_text7
5
3%26 = phi float [ 0.000000e+00, %17 ], [ %37, %29 ]
)float8B

	full_text

	float %37
Lstore8BA
?
	full_text2
0
.store float %26, float* %14, align 4, !tbaa !8
)float8B

	full_text

	float %26
+float*8B

	full_text


float* %14
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
:br8B2
0
	full_text#
!
br i1 %15, label %41, label %27
#i18B

	full_text


i1 %15
9and8B0
.
	full_text!

%28 = and i64 %19, 4294967295
%i648B

	full_text
	
i64 %19
'br8B

	full_text

br label %59
Dphi8B;
9
	full_text,
*
(%30 = phi i64 [ %11, %23 ], [ %38, %29 ]
%i648B

	full_text
	
i64 %11
%i648B

	full_text
	
i64 %38
Ophi8BF
D
	full_text7
5
3%31 = phi float [ 0.000000e+00, %23 ], [ %37, %29 ]
)float8B

	full_text

	float %37
9and8B0
.
	full_text!

%32 = and i64 %30, 4294967295
%i648B

	full_text
	
i64 %30
]getelementptr8BJ
H
	full_text;
9
7%33 = getelementptr inbounds float, float* %22, i64 %32
+float*8B

	full_text


float* %22
%i648B

	full_text
	
i64 %32
Lload8BB
@
	full_text3
1
/%34 = load float, float* %33, align 4, !tbaa !8
+float*8B

	full_text


float* %33
\getelementptr8BI
G
	full_text:
8
6%35 = getelementptr inbounds float, float* %1, i64 %32
%i648B

	full_text
	
i64 %32
Lload8BB
@
	full_text3
1
/%36 = load float, float* %35, align 4, !tbaa !8
+float*8B

	full_text


float* %35
ecall8B[
Y
	full_textL
J
H%37 = tail call float @llvm.fmuladd.f32(float %34, float %36, float %31)
)float8B

	full_text

	float %34
)float8B

	full_text

	float %36
)float8B

	full_text

	float %31
2add8B)
'
	full_text

%38 = add i64 %24, %32
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %32
8trunc8B-
+
	full_text

%39 = trunc i64 %38 to i32
%i648B

	full_text
	
i64 %38
7icmp8B-
+
	full_text

%40 = icmp ult i32 %39, %2
%i328B

	full_text
	
i32 %39
:br8B2
0
	full_text#
!
br i1 %40, label %29, label %25
#i18B

	full_text


i1 %40
Ocall8BE
C
	full_text6
4
2%42 = tail call i64 @_Z14get_local_sizej(i32 0) #4
5icmp8B+
)
	full_text

%43 = icmp eq i64 %42, 0
%i648B

	full_text
	
i64 %42
:br8B2
0
	full_text#
!
br i1 %43, label %45, label %44
#i18B

	full_text


i1 %43
'br8	B

	full_text

br label %49
Ophi8
BF
D
	full_text7
5
3%46 = phi float [ 0.000000e+00, %41 ], [ %55, %49 ]
)float8
B

	full_text

	float %55
9and8
B0
.
	full_text!

%47 = and i64 %19, 4294967295
%i648
B

	full_text
	
i64 %19
\getelementptr8
BI
G
	full_text:
8
6%48 = getelementptr inbounds float, float* %4, i64 %47
%i648
B

	full_text
	
i64 %47
Lstore8
BA
?
	full_text2
0
.store float %46, float* %48, align 4, !tbaa !8
)float8
B

	full_text

	float %46
+float*8
B

	full_text


float* %48
'br8
B

	full_text

br label %59
Bphi8B9
7
	full_text*
(
&%50 = phi i64 [ %57, %49 ], [ 0, %44 ]
%i648B

	full_text
	
i64 %57
Bphi8B9
7
	full_text*
(
&%51 = phi i32 [ %56, %49 ], [ 0, %44 ]
%i328B

	full_text
	
i32 %56
Ophi8BF
D
	full_text7
5
3%52 = phi float [ %55, %49 ], [ 0.000000e+00, %44 ]
)float8B

	full_text

	float %55
\getelementptr8BI
G
	full_text:
8
6%53 = getelementptr inbounds float, float* %5, i64 %50
%i648B

	full_text
	
i64 %50
Lload8BB
@
	full_text3
1
/%54 = load float, float* %53, align 4, !tbaa !8
+float*8B

	full_text


float* %53
6fadd8B,
*
	full_text

%55 = fadd float %52, %54
)float8B

	full_text

	float %52
)float8B

	full_text

	float %54
0add8B'
%
	full_text

%56 = add i32 %51, 1
%i328B

	full_text
	
i32 %51
6zext8B,
*
	full_text

%57 = zext i32 %56 to i64
%i328B

	full_text
	
i32 %56
8icmp8B.
,
	full_text

%58 = icmp ugt i64 %42, %57
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %57
:br8B2
0
	full_text#
!
br i1 %58, label %49, label %45
#i18B

	full_text


i1 %58
Dphi8B;
9
	full_text,
*
(%60 = phi i64 [ %28, %27 ], [ %47, %45 ]
%i648B

	full_text
	
i64 %28
%i648B

	full_text
	
i64 %47
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
Ocall8BE
C
	full_text6
4
2%61 = tail call i64 @_Z14get_num_groupsj(i32 0) #4
2add8B)
'
	full_text

%62 = add i64 %61, %60
%i648B

	full_text
	
i64 %61
%i648B

	full_text
	
i64 %60
8trunc8B-
+
	full_text

%63 = trunc i64 %62 to i32
%i648B

	full_text
	
i64 %62
7icmp8B-
+
	full_text

%64 = icmp ult i32 %63, %3
%i328B

	full_text
	
i32 %63
:br8B2
0
	full_text#
!
br i1 %64, label %17, label %16
#i18B

	full_text


i1 %64
*float*8B

	full_text

	float* %0
*float*8B

	full_text

	float* %4
$i328B

	full_text


i32 %2
*float*8B

	full_text

	float* %5
$i328B

	full_text


i32 %3
*float*8B
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
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
#i648B

	full_text	

i64 0
2float8B%
#
	full_text

float 0.000000e+00
,i648B!

	full_text

i64 4294967295
#i328B

	full_text	

i32 1
#i328B

	full_text	

i32 0       	
 		                     ! "$ ## %& %' %% (( )* ), ++ -/ .0 .. 12 11 34 33 56 57 55 89 88 :; :: <= << >? >@ >A >> BC BD BB EF EE GH GG IJ IK LM LL NO NR QQ ST SS UV UU WX WY WW Z\ [[ ]^ ]] _` __ ab aa cd cc ef eg ee hi hh jk jj lm ln ll op or qs qq tt uu vw vx vv yz yy {| {{ }~ } ? U	? 	? 	? G? ? a	? 	? {? :    
	    y  v      > $# & ' * , /B 0> 2. 4 63 75 93 ;: =8 ?< @1 A! C3 DB FE HG JK ML Oe R TS VQ XU Yj \h ^e `[ ba d_ fc g] ih kK mj nl p+ rS su wq xv zy |{ ~    ! #" .) K) +I .I #N QN P- qZ qP [} } o [o Q ??  ?? ?? ?? ?? ?? ?? > ?? > ?? t ?? tu ?? u! ?? !K ?? K( ?? (	? 	? L	? [? #? 1? Q	? _	? +	? 3	? S? (	? h? t? ? ? !? K	? ]? u"
MatVecMulCoalesced0"
_Z12get_group_idj"
_Z12get_local_idj"
llvm.fmuladd.f32"
_Z14get_local_sizej"
_Z7barrierj"
_Z14get_num_groupsj*?
+nvidia-4.2-MatVecMul-MatVecMulCoalesced0.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?

devmap_label


wgsize
?

transfer_bytes	
????

wgsize_log1p
?9?A
 
transfer_bytes_log1p
?9?A