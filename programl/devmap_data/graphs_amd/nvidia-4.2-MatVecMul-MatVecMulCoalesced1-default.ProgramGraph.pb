
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
 
br i1 %9, label %10, label %19
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
?bitcast8B2
0
	full_text#
!
%16 = bitcast float* %5 to i32*
Ocall8BE
C
	full_text6
4
2%17 = tail call i64 @_Z14get_local_sizej(i32 0) #4
6icmp8B,
*
	full_text

%18 = icmp ugt i64 %17, 1
%i648B

	full_text
	
i64 %17
'br8B

	full_text

br label %20
$ret8B

	full_text


ret void
Cphi8B:
8
	full_text+
)
'%21 = phi i32 [ %8, %10 ], [ %72, %68 ]
$i328B

	full_text


i32 %8
%i328B

	full_text
	
i32 %72
Cphi8B:
8
	full_text+
)
'%22 = phi i64 [ %7, %10 ], [ %71, %68 ]
$i648B

	full_text


i64 %7
%i648B

	full_text
	
i64 %71
1mul8B(
&
	full_text

%23 = mul i32 %21, %2
%i328B

	full_text
	
i32 %21
6zext8B,
*
	full_text

%24 = zext i32 %23 to i64
%i328B

	full_text
	
i32 %23
\getelementptr8BI
G
	full_text:
8
6%25 = getelementptr inbounds float, float* %0, i64 %24
%i648B

	full_text
	
i64 %24
:br8B2
0
	full_text#
!
br i1 %13, label %26, label %27
#i18B

	full_text


i1 %13
'br8B

	full_text

br label %30
Ophi8BF
D
	full_text7
5
3%28 = phi float [ 0.000000e+00, %20 ], [ %38, %30 ]
)float8B

	full_text

	float %38
Lstore8BA
?
	full_text2
0
.store float %28, float* %14, align 4, !tbaa !8
)float8B

	full_text

	float %28
+float*8B

	full_text


float* %14
:br8B2
0
	full_text#
!
br i1 %18, label %29, label %42
#i18B

	full_text


i1 %18
'br8B

	full_text

br label %45
Dphi8B;
9
	full_text,
*
(%31 = phi i64 [ %39, %30 ], [ %11, %26 ]
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %11
Ophi8BF
D
	full_text7
5
3%32 = phi float [ %38, %30 ], [ 0.000000e+00, %26 ]
)float8B

	full_text

	float %38
9and8B0
.
	full_text!

%33 = and i64 %31, 4294967295
%i648B

	full_text
	
i64 %31
]getelementptr8BJ
H
	full_text;
9
7%34 = getelementptr inbounds float, float* %25, i64 %33
+float*8B

	full_text


float* %25
%i648B

	full_text
	
i64 %33
Lload8BB
@
	full_text3
1
/%35 = load float, float* %34, align 4, !tbaa !8
+float*8B

	full_text


float* %34
\getelementptr8BI
G
	full_text:
8
6%36 = getelementptr inbounds float, float* %1, i64 %33
%i648B

	full_text
	
i64 %33
Lload8BB
@
	full_text3
1
/%37 = load float, float* %36, align 4, !tbaa !8
+float*8B

	full_text


float* %36
ecall8B[
Y
	full_textL
J
H%38 = tail call float @llvm.fmuladd.f32(float %35, float %37, float %32)
)float8B

	full_text

	float %35
)float8B

	full_text

	float %37
)float8B

	full_text

	float %32
2add8B)
'
	full_text

%39 = add i64 %17, %33
%i648B

	full_text
	
i64 %17
%i648B

	full_text
	
i64 %33
8trunc8B-
+
	full_text

%40 = trunc i64 %39 to i32
%i648B

	full_text
	
i64 %39
7icmp8B-
+
	full_text

%41 = icmp ult i32 %40, %2
%i328B

	full_text
	
i32 %40
:br8B2
0
	full_text#
!
br i1 %41, label %30, label %27
#i18B

	full_text


i1 %41
:br8B2
0
	full_text#
!
br i1 %15, label %63, label %43
#i18B

	full_text


i1 %15
9and8	B0
.
	full_text!

%44 = and i64 %22, 4294967295
%i648	B

	full_text
	
i64 %22
'br8	B

	full_text

br label %68
Bphi8
B9
7
	full_text*
(
&%46 = phi i32 [ %47, %61 ], [ 1, %29 ]
%i328
B

	full_text
	
i32 %47
Bcall8
B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
0shl8
B'
%
	full_text

%47 = shl i32 %46, 1
%i328
B

	full_text
	
i32 %46
6zext8
B,
*
	full_text

%48 = zext i32 %47 to i64
%i328
B

	full_text
	
i32 %47
2mul8
B)
'
	full_text

%49 = mul i64 %11, %48
%i648
B

	full_text
	
i64 %11
%i648
B

	full_text
	
i64 %48
9and8
B0
.
	full_text!

%50 = and i64 %49, 4294967294
%i648
B

	full_text
	
i64 %49
8icmp8
B.
,
	full_text

%51 = icmp ult i64 %50, %17
%i648
B

	full_text
	
i64 %50
%i648
B

	full_text
	
i64 %17
:br8
B2
0
	full_text#
!
br i1 %51, label %52, label %61
#i18
B

	full_text


i1 %51
8trunc8B-
+
	full_text

%53 = trunc i64 %49 to i32
%i648B

	full_text
	
i64 %49
2add8B)
'
	full_text

%54 = add i32 %46, %53
%i328B

	full_text
	
i32 %46
%i328B

	full_text
	
i32 %53
6zext8B,
*
	full_text

%55 = zext i32 %54 to i64
%i328B

	full_text
	
i32 %54
\getelementptr8BI
G
	full_text:
8
6%56 = getelementptr inbounds float, float* %5, i64 %55
%i648B

	full_text
	
i64 %55
Lload8BB
@
	full_text3
1
/%57 = load float, float* %56, align 4, !tbaa !8
+float*8B

	full_text


float* %56
\getelementptr8BI
G
	full_text:
8
6%58 = getelementptr inbounds float, float* %5, i64 %50
%i648B

	full_text
	
i64 %50
Lload8BB
@
	full_text3
1
/%59 = load float, float* %58, align 4, !tbaa !8
+float*8B

	full_text


float* %58
6fadd8B,
*
	full_text

%60 = fadd float %57, %59
)float8B

	full_text

	float %57
)float8B

	full_text

	float %59
Lstore8BA
?
	full_text2
0
.store float %60, float* %58, align 4, !tbaa !8
)float8B

	full_text

	float %60
+float*8B

	full_text


float* %58
'br8B

	full_text

br label %61
8icmp8B.
,
	full_text

%62 = icmp ugt i64 %17, %48
%i648B

	full_text
	
i64 %17
%i648B

	full_text
	
i64 %48
:br8B2
0
	full_text#
!
br i1 %62, label %45, label %42
#i18B

	full_text


i1 %62
Hload8B>
<
	full_text/
-
+%64 = load i32, i32* %16, align 4, !tbaa !8
'i32*8B

	full_text


i32* %16
9and8B0
.
	full_text!

%65 = and i64 %22, 4294967295
%i648B

	full_text
	
i64 %22
\getelementptr8BI
G
	full_text:
8
6%66 = getelementptr inbounds float, float* %4, i64 %65
%i648B

	full_text
	
i64 %65
@bitcast8B3
1
	full_text$
"
 %67 = bitcast float* %66 to i32*
+float*8B

	full_text


float* %66
Hstore8B=
;
	full_text.
,
*store i32 %64, i32* %67, align 4, !tbaa !8
%i328B

	full_text
	
i32 %64
'i32*8B

	full_text


i32* %67
'br8B

	full_text

br label %68
Dphi8B;
9
	full_text,
*
(%69 = phi i64 [ %44, %43 ], [ %65, %63 ]
%i648B

	full_text
	
i64 %44
%i648B

	full_text
	
i64 %65
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
Ocall8BE
C
	full_text6
4
2%70 = tail call i64 @_Z14get_num_groupsj(i32 0) #4
2add8B)
'
	full_text

%71 = add i64 %70, %69
%i648B

	full_text
	
i64 %70
%i648B

	full_text
	
i64 %69
8trunc8B-
+
	full_text

%72 = trunc i64 %71 to i32
%i648B

	full_text
	
i64 %71
7icmp8B-
+
	full_text

%73 = icmp ult i32 %72, %3
%i328B

	full_text
	
i32 %72
:br8B2
0
	full_text#
!
br i1 %73, label %20, label %19
#i18B

	full_text


i1 %73
$i328B

	full_text


i32 %3
$i328B

	full_text


i32 %2
*float*8B

	full_text

	float* %4
*float*8B

	full_text

	float* %5
*float*8B

	full_text

	float* %1
*float*8B

	full_text

	float* %0
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
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 0
#i328B

	full_text	

i32 0
2float8B%
#
	full_text

float 0.000000e+00
,i648B!

	full_text

i64 4294967295
#i328B

	full_text	

i32 1
,i648B!

	full_text

i64 4294967294       	
 		                      !" !! #$ #' && () (* (( +, +/ .0 .. 12 11 34 33 56 57 55 89 88 :; :: <= << >? >@ >A >> BC BD BB EF EE GH GG IJ IL KN MM OQ PP RR ST SS UV UU WX WY WW Z[ ZZ \] \^ \\ _` _b aa cd ce cc fg ff hi hh jk jj lm ll no nn pq pr pp st su ss vx wy ww z{ z} || ~ ~~ Ä
Å ÄÄ ÇÉ ÇÇ ÑÖ Ñ
Ü ÑÑ áâ à
ä àà ãã åå çé ç
è çç êë êê íì íí îï î	ñ 
ñ í	ó 	ó 	ó Gò Äô ô ô hô lö :õ !    
	     ê  ç     " $> '& ) * ,B / 0> 2. 4! 63 75 93 ;: =8 ?< @1 A C3 DB FE HG J L NS QP TS V XU YW [Z ] ^\ `W bP da ec gf ih kZ ml oj qn rp tl u xU yw { } ~ ÅÄ É| ÖÇ ÜM â~ äå éà èç ëê ìí ï   # %# &% .+ -+ KI .I &- PK |K M_ a_ wá àO àv wz Pz Kî î  úú ùù üü ûû ††  °° üü > ûû > úú  ùù R †† Rå °° åã †† ã	¢ 	£ § § § § å• &	• 1	¶ 3	¶ M	¶ ~	ß Pß R	ß Sß ã	® Z"
MatVecMulCoalesced1"
_Z12get_group_idj"
_Z12get_local_idj"
llvm.fmuladd.f32"
_Z14get_local_sizej"
_Z7barrierj"
_Z14get_num_groupsj*§
+nvidia-4.2-MatVecMul-MatVecMulCoalesced1.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Ç

devmap_label


transfer_bytes	
∞ìÄ“

wgsize_log1p
√9üA

wgsize
Ä
 
transfer_bytes_log1p
√9üA