

[external]
JcallBB
@
	full_text3
1
/%8 = tail call i64 @_Z12get_local_idj(i32 0) #4
4truncB+
)
	full_text

%9 = trunc i64 %8 to i32
"i64B

	full_text


i64 %8
2addB+
)
	full_text

%10 = add nsw i32 %5, -1
/andB(
&
	full_text

%11 = and i32 %10, %9
#i32B

	full_text
	
i32 %10
"i32B

	full_text


i32 %9
McallBE
C
	full_text6
4
2%12 = tail call i64 @_Z14get_local_sizej(i32 0) #4
3sextB+
)
	full_text

%13 = sext i32 %5 to i64
2udivB*
(
	full_text

%14 = udiv i64 %12, %13
#i64B

	full_text
	
i64 %12
#i64B

	full_text
	
i64 %13
KcallBC
A
	full_text4
2
0%15 = tail call i64 @_Z12get_group_idj(i32 0) #4
/shlB(
&
	full_text

%16 = shl i64 %14, 32
#i64B

	full_text
	
i64 %14
7ashrB/
-
	full_text 

%17 = ashr exact i64 %16, 32
#i64B

	full_text
	
i64 %16
0mulB)
'
	full_text

%18 = mul i64 %17, %15
#i64B

	full_text
	
i64 %17
#i64B

	full_text
	
i64 %15
0sdivB(
&
	full_text

%19 = sdiv i32 %9, %5
"i32B

	full_text


i32 %9
4zextB,
*
	full_text

%20 = zext i32 %19 to i64
#i32B

	full_text
	
i32 %19
0addB)
'
	full_text

%21 = add i64 %18, %20
#i64B

	full_text
	
i64 %18
#i64B

	full_text
	
i64 %20
6truncB-
+
	full_text

%22 = trunc i64 %21 to i32
#i64B

	full_text
	
i64 %21
.shlB'
%
	full_text

%23 = shl i64 %8, 32
"i64B

	full_text


i64 %8
7ashrB/
-
	full_text 

%24 = ashr exact i64 %23, 32
#i64B

	full_text
	
i64 %23
ìgetelementptrBÅ

	full_textr
p
n%25 = getelementptr inbounds [128 x float], [128 x float]* @spmv_csr_vector_kernel.partialSums, i64 0, i64 %24
#i64B

	full_text
	
i64 %24
\storeBS
Q
	full_textD
B
@store volatile float 0.000000e+00, float* %25, align 4, !tbaa !8
)float*B

	full_text


float* %25
5icmpB-
+
	full_text

%26 = icmp slt i32 %22, %4
#i32B

	full_text
	
i32 %22
8brB2
0
	full_text#
!
br i1 %26, label %27, label %76
!i1B

	full_text


i1 %26
1shl8B(
&
	full_text

%28 = shl i64 %21, 32
%i648B

	full_text
	
i64 %21
9ashr8B/
-
	full_text 

%29 = ashr exact i64 %28, 32
%i648B

	full_text
	
i64 %28
Xgetelementptr8BE
C
	full_text6
4
2%30 = getelementptr inbounds i32, i32* %3, i64 %29
%i648B

	full_text
	
i64 %29
Iload8B?
=
	full_text0
.
,%31 = load i32, i32* %30, align 4, !tbaa !12
'i32*8B

	full_text


i32* %30
9add8B0
.
	full_text!

%32 = add i64 %28, 4294967296
%i648B

	full_text
	
i64 %28
9ashr8B/
-
	full_text 

%33 = ashr exact i64 %32, 32
%i648B

	full_text
	
i64 %32
Xgetelementptr8BE
C
	full_text6
4
2%34 = getelementptr inbounds i32, i32* %3, i64 %33
%i648B

	full_text
	
i64 %33
Iload8B?
=
	full_text0
.
,%35 = load i32, i32* %34, align 4, !tbaa !12
'i32*8B

	full_text


i32* %34
6add8B-
+
	full_text

%36 = add nsw i32 %31, %11
%i328B

	full_text
	
i32 %31
%i328B

	full_text
	
i32 %11
8icmp8B.
,
	full_text

%37 = icmp slt i32 %36, %35
%i328B

	full_text
	
i32 %36
%i328B

	full_text
	
i32 %35
:br8B2
0
	full_text#
!
br i1 %37, label %38, label %41
#i18B

	full_text


i1 %37
6sext8B,
*
	full_text

%39 = sext i32 %36 to i64
%i328B

	full_text
	
i32 %36
6sext8B,
*
	full_text

%40 = sext i32 %35 to i64
%i328B

	full_text
	
i32 %35
'br8B

	full_text

br label %45
Ophi8BF
D
	full_text7
5
3%42 = phi float [ 0.000000e+00, %27 ], [ %55, %45 ]
)float8B

	full_text

	float %55
Ustore8BJ
H
	full_text;
9
7store volatile float %42, float* %25, align 4, !tbaa !8
)float8B

	full_text

	float %42
+float*8B

	full_text


float* %25
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
5icmp8B+
)
	full_text

%43 = icmp sgt i32 %5, 1
:br8B2
0
	full_text#
!
br i1 %43, label %44, label %71
#i18B

	full_text


i1 %43
'br8B

	full_text

br label %58
Dphi8B;
9
	full_text,
*
(%46 = phi i64 [ %39, %38 ], [ %56, %45 ]
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %56
Ophi8BF
D
	full_text7
5
3%47 = phi float [ 0.000000e+00, %38 ], [ %55, %45 ]
)float8B

	full_text

	float %55
Xgetelementptr8BE
C
	full_text6
4
2%48 = getelementptr inbounds i32, i32* %2, i64 %46
%i648B

	full_text
	
i64 %46
Iload8B?
=
	full_text0
.
,%49 = load i32, i32* %48, align 4, !tbaa !12
'i32*8B

	full_text


i32* %48
\getelementptr8BI
G
	full_text:
8
6%50 = getelementptr inbounds float, float* %0, i64 %46
%i648B

	full_text
	
i64 %46
Lload8BB
@
	full_text3
1
/%51 = load float, float* %50, align 4, !tbaa !8
+float*8B

	full_text


float* %50
6sext8B,
*
	full_text

%52 = sext i32 %49 to i64
%i328B

	full_text
	
i32 %49
\getelementptr8BI
G
	full_text:
8
6%53 = getelementptr inbounds float, float* %1, i64 %52
%i648B

	full_text
	
i64 %52
Lload8BB
@
	full_text3
1
/%54 = load float, float* %53, align 4, !tbaa !8
+float*8B

	full_text


float* %53
ecall8B[
Y
	full_textL
J
H%55 = tail call float @llvm.fmuladd.f32(float %51, float %54, float %47)
)float8B

	full_text

	float %51
)float8B

	full_text

	float %54
)float8B

	full_text

	float %47
2add8B)
'
	full_text

%56 = add i64 %46, %13
%i648B

	full_text
	
i64 %46
%i648B

	full_text
	
i64 %13
8icmp8B.
,
	full_text

%57 = icmp slt i64 %56, %40
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %40
:br8B2
0
	full_text#
!
br i1 %57, label %45, label %41
#i18B

	full_text


i1 %57
Cphi8B:
8
	full_text+
)
'%59 = phi i32 [ %60, %69 ], [ %5, %44 ]
%i328B

	full_text
	
i32 %60
2sdiv8B(
&
	full_text

%60 = sdiv i32 %59, 2
%i328B

	full_text
	
i32 %59
8icmp8B.
,
	full_text

%61 = icmp slt i32 %11, %60
%i328B

	full_text
	
i32 %11
%i328B

	full_text
	
i32 %60
:br8B2
0
	full_text#
!
br i1 %61, label %62, label %69
#i18B

	full_text


i1 %61
5add8B,
*
	full_text

%63 = add nsw i32 %60, %9
%i328B

	full_text
	
i32 %60
$i328B

	full_text


i32 %9
6sext8B,
*
	full_text

%64 = sext i32 %63 to i64
%i328B

	full_text
	
i32 %63
ïgetelementptr8BÅ

	full_textr
p
n%65 = getelementptr inbounds [128 x float], [128 x float]* @spmv_csr_vector_kernel.partialSums, i64 0, i64 %64
%i648B

	full_text
	
i64 %64
Uload8BK
I
	full_text<
:
8%66 = load volatile float, float* %65, align 4, !tbaa !8
+float*8B

	full_text


float* %65
Uload8BK
I
	full_text<
:
8%67 = load volatile float, float* %25, align 4, !tbaa !8
+float*8B

	full_text


float* %25
6fadd8B,
*
	full_text

%68 = fadd float %66, %67
)float8B

	full_text

	float %66
)float8B

	full_text

	float %67
Ustore8BJ
H
	full_text;
9
7store volatile float %68, float* %25, align 4, !tbaa !8
)float8B

	full_text

	float %68
+float*8B

	full_text


float* %25
'br8B

	full_text

br label %69
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
6icmp8B,
*
	full_text

%70 = icmp sgt i32 %59, 3
%i328B

	full_text
	
i32 %59
:br8B2
0
	full_text#
!
br i1 %70, label %58, label %71
#i18B

	full_text


i1 %70
5icmp8	B+
)
	full_text

%72 = icmp eq i32 %11, 0
%i328	B

	full_text
	
i32 %11
:br8	B2
0
	full_text#
!
br i1 %72, label %73, label %76
#i18	B

	full_text


i1 %72
Uload8
BK
I
	full_text<
:
8%74 = load volatile float, float* %25, align 4, !tbaa !8
+float*8
B

	full_text


float* %25
\getelementptr8
BI
G
	full_text:
8
6%75 = getelementptr inbounds float, float* %6, i64 %29
%i648
B

	full_text
	
i64 %29
Lstore8
BA
?
	full_text2
0
.store float %74, float* %75, align 4, !tbaa !8
)float8
B

	full_text

	float %74
+float*8
B

	full_text


float* %75
'br8
B

	full_text

br label %76
$ret8B

	full_text


ret void
*float*8B

	full_text

	float* %1
*float*8B

	full_text

	float* %6
&i32*8B

	full_text
	
i32* %3
&i32*8B

	full_text
	
i32* %2
*float*8B

	full_text

	float* %0
$i328B

	full_text


i32 %5
$i328B

	full_text


i32 %4
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
|[128 x float]*8Bf
d
	full_textW
U
S@spmv_csr_vector_kernel.partialSums = internal global [128 x float] undef, align 16
$i328B

	full_text


i32 -1
#i648B

	full_text	

i64 0
#i328B

	full_text	

i32 0
#i328B

	full_text	

i32 2
#i328B

	full_text	

i32 3
#i328B

	full_text	

i32 1
$i648B

	full_text


i64 32
,i648B!

	full_text

i64 4294967296
2float8B%
#
	full_text

float 0.000000e+00        		 
 
 

                     !    "# "" $% $$ &' && () (+ ** ,- ,, ./ .. 01 00 23 22 45 44 67 66 89 88 :; :< :: => =? == @A @C BB DE DD FH GG IJ IK II LL MM NO NR QS QQ TU TT VW VV XY XX Z[ ZZ \] \\ ^_ ^^ `a `` bc bb de df dg dd hi hj hh kl km kk no nq pp rs rr tu tv tt wx wz y{ yy |} || ~ ~~ ÄÅ ÄÄ ÇÉ ÇÇ ÑÖ Ñ
Ü ÑÑ áà á
â áá äã åç åå éè éë êê íì íï îî ñ
ó ññ òô ò
ö òò õù `û ñü .ü 6† V° Z¢ ¢ 		¢ ¢ M	¢ p	£ &    	 
           !  #" % '& ) +* -, /. 1* 32 54 76 90 ; <: >8 ?= A: C8 Ed HG J" KM OB Rh Sd UQ WV YQ [Z ]X _^ a` c\ eb fT gQ i	 jh lD mk or qp s ur vt xr z {y }| ~ Å" ÉÄ ÖÇ ÜÑ à" âp çå è ëê ì" ï, óî ôñ ö( *( ú@ B@ GF QN PN ên Qn GP pí îí úw yw ãõ úä ãé pé ê ú •• ßß §§ ¶¶ ®®ã ®® ã §§  •• L ®® L ¶¶ d ßß d© "© ~	™ 	´ "	´ ~¨ ¨ ¨ 
¨ ê	≠ r
Æ åØ L	Ø MØ ã	∞ 	∞ 	∞ 	∞  	∞ *	∞ ,	∞ 4	± 2≤ $≤ G≤ T"
spmv_csr_vector_kernel"
_Z12get_local_idj"
_Z14get_local_sizej"
_Z12get_group_idj"
llvm.fmuladd.f32"
_Z7barrierj*¢
)shoc-1.1.5-Spmv-spmv_csr_vector_kernel.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Ä

wgsize
Ä

transfer_bytes
Ùçz

wgsize_log1p
¿$hA
 
transfer_bytes_log1p
¿$hA

devmap_label
 