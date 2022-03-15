

[external]
KcallBC
A
	full_text4
2
0%6 = tail call i64 @_Z13get_global_idj(i32 0) #3
4truncB+
)
	full_text

%7 = trunc i64 %6 to i32
"i64B

	full_text


i64 %6
3icmpB+
)
	full_text

%8 = icmp slt i32 %7, %3
"i32B

	full_text


i32 %7
6brB0
.
	full_text!

br i1 %8, label %9, label %75
 i1B

	full_text	

i1 %8
0shl8B'
%
	full_text

%10 = shl i64 %6, 32
$i648B

	full_text


i64 %6
9ashr8B/
-
	full_text 

%11 = ashr exact i64 %10, 32
%i648B

	full_text
	
i64 %10
\getelementptr8BI
G
	full_text:
8
6%12 = getelementptr inbounds float, float* %2, i64 %11
%i648B

	full_text
	
i64 %11
Ustore8BJ
H
	full_text;
9
7store float 0.000000e+00, float* %12, align 4, !tbaa !9
+float*8B

	full_text


float* %12
5icmp8B+
)
	full_text

%13 = icmp sgt i32 %4, 0
:br8B2
0
	full_text#
!
br i1 %13, label %14, label %75
#i18B

	full_text


i1 %13
4mul8B+
)
	full_text

%15 = mul nsw i32 %7, %4
$i328B

	full_text


i32 %7
6sext8B,
*
	full_text

%16 = sext i32 %15 to i64
%i328B

	full_text
	
i32 %15
5zext8B+
)
	full_text

%17 = zext i32 %4 to i64
5add8B,
*
	full_text

%18 = add nsw i64 %17, -1
%i648B

	full_text
	
i64 %17
0and8B'
%
	full_text

%19 = and i64 %17, 3
%i648B

	full_text
	
i64 %17
6icmp8B,
*
	full_text

%20 = icmp ult i64 %18, 3
%i648B

	full_text
	
i64 %18
:br8B2
0
	full_text#
!
br i1 %20, label %57, label %21
#i18B

	full_text


i1 %20
6sub8B-
+
	full_text

%22 = sub nsw i64 %17, %19
%i648B

	full_text
	
i64 %17
%i648B

	full_text
	
i64 %19
'br8B

	full_text

br label %23
Ophi8BF
D
	full_text7
5
3%24 = phi float [ 0.000000e+00, %21 ], [ %53, %23 ]
)float8B

	full_text

	float %53
Bphi8B9
7
	full_text*
(
&%25 = phi i64 [ 0, %21 ], [ %54, %23 ]
%i648B

	full_text
	
i64 %54
Dphi8B;
9
	full_text,
*
(%26 = phi i64 [ %22, %21 ], [ %55, %23 ]
%i648B

	full_text
	
i64 %22
%i648B

	full_text
	
i64 %55
6add8B-
+
	full_text

%27 = add nsw i64 %25, %16
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %16
\getelementptr8BI
G
	full_text:
8
6%28 = getelementptr inbounds float, float* %0, i64 %27
%i648B

	full_text
	
i64 %27
Lload8BB
@
	full_text3
1
/%29 = load float, float* %28, align 4, !tbaa !9
+float*8B

	full_text


float* %28
\getelementptr8BI
G
	full_text:
8
6%30 = getelementptr inbounds float, float* %1, i64 %25
%i648B

	full_text
	
i64 %25
Lload8BB
@
	full_text3
1
/%31 = load float, float* %30, align 4, !tbaa !9
+float*8B

	full_text


float* %30
ecall8B[
Y
	full_textL
J
H%32 = tail call float @llvm.fmuladd.f32(float %29, float %31, float %24)
)float8B

	full_text

	float %29
)float8B

	full_text

	float %31
)float8B

	full_text

	float %24
Lstore8BA
?
	full_text2
0
.store float %32, float* %12, align 4, !tbaa !9
)float8B

	full_text

	float %32
+float*8B

	full_text


float* %12
.or8B&
$
	full_text

%33 = or i64 %25, 1
%i648B

	full_text
	
i64 %25
6add8B-
+
	full_text

%34 = add nsw i64 %33, %16
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %16
\getelementptr8BI
G
	full_text:
8
6%35 = getelementptr inbounds float, float* %0, i64 %34
%i648B

	full_text
	
i64 %34
Lload8BB
@
	full_text3
1
/%36 = load float, float* %35, align 4, !tbaa !9
+float*8B

	full_text


float* %35
\getelementptr8BI
G
	full_text:
8
6%37 = getelementptr inbounds float, float* %1, i64 %33
%i648B

	full_text
	
i64 %33
Lload8BB
@
	full_text3
1
/%38 = load float, float* %37, align 4, !tbaa !9
+float*8B

	full_text


float* %37
ecall8B[
Y
	full_textL
J
H%39 = tail call float @llvm.fmuladd.f32(float %36, float %38, float %32)
)float8B

	full_text

	float %36
)float8B

	full_text

	float %38
)float8B

	full_text

	float %32
Lstore8BA
?
	full_text2
0
.store float %39, float* %12, align 4, !tbaa !9
)float8B

	full_text

	float %39
+float*8B

	full_text


float* %12
.or8B&
$
	full_text

%40 = or i64 %25, 2
%i648B

	full_text
	
i64 %25
6add8B-
+
	full_text

%41 = add nsw i64 %40, %16
%i648B

	full_text
	
i64 %40
%i648B

	full_text
	
i64 %16
\getelementptr8BI
G
	full_text:
8
6%42 = getelementptr inbounds float, float* %0, i64 %41
%i648B

	full_text
	
i64 %41
Lload8BB
@
	full_text3
1
/%43 = load float, float* %42, align 4, !tbaa !9
+float*8B

	full_text


float* %42
\getelementptr8BI
G
	full_text:
8
6%44 = getelementptr inbounds float, float* %1, i64 %40
%i648B

	full_text
	
i64 %40
Lload8BB
@
	full_text3
1
/%45 = load float, float* %44, align 4, !tbaa !9
+float*8B

	full_text


float* %44
ecall8B[
Y
	full_textL
J
H%46 = tail call float @llvm.fmuladd.f32(float %43, float %45, float %39)
)float8B

	full_text

	float %43
)float8B

	full_text

	float %45
)float8B

	full_text

	float %39
Lstore8BA
?
	full_text2
0
.store float %46, float* %12, align 4, !tbaa !9
)float8B

	full_text

	float %46
+float*8B

	full_text


float* %12
.or8B&
$
	full_text

%47 = or i64 %25, 3
%i648B

	full_text
	
i64 %25
6add8B-
+
	full_text

%48 = add nsw i64 %47, %16
%i648B

	full_text
	
i64 %47
%i648B

	full_text
	
i64 %16
\getelementptr8BI
G
	full_text:
8
6%49 = getelementptr inbounds float, float* %0, i64 %48
%i648B

	full_text
	
i64 %48
Lload8BB
@
	full_text3
1
/%50 = load float, float* %49, align 4, !tbaa !9
+float*8B

	full_text


float* %49
\getelementptr8BI
G
	full_text:
8
6%51 = getelementptr inbounds float, float* %1, i64 %47
%i648B

	full_text
	
i64 %47
Lload8BB
@
	full_text3
1
/%52 = load float, float* %51, align 4, !tbaa !9
+float*8B

	full_text


float* %51
ecall8B[
Y
	full_textL
J
H%53 = tail call float @llvm.fmuladd.f32(float %50, float %52, float %46)
)float8B

	full_text

	float %50
)float8B

	full_text

	float %52
)float8B

	full_text

	float %46
Lstore8BA
?
	full_text2
0
.store float %53, float* %12, align 4, !tbaa !9
)float8B

	full_text

	float %53
+float*8B

	full_text


float* %12
4add8B+
)
	full_text

%54 = add nsw i64 %25, 4
%i648B

	full_text
	
i64 %25
1add8B(
&
	full_text

%55 = add i64 %26, -4
%i648B

	full_text
	
i64 %26
5icmp8B+
)
	full_text

%56 = icmp eq i64 %55, 0
%i648B

	full_text
	
i64 %55
:br8B2
0
	full_text#
!
br i1 %56, label %57, label %23
#i18B

	full_text


i1 %56
Ophi8BF
D
	full_text7
5
3%58 = phi float [ 0.000000e+00, %14 ], [ %53, %23 ]
)float8B

	full_text

	float %53
Bphi8B9
7
	full_text*
(
&%59 = phi i64 [ 0, %14 ], [ %54, %23 ]
%i648B

	full_text
	
i64 %54
5icmp8B+
)
	full_text

%60 = icmp eq i64 %19, 0
%i648B

	full_text
	
i64 %19
:br8B2
0
	full_text#
!
br i1 %60, label %75, label %61
#i18B

	full_text


i1 %60
'br8B

	full_text

br label %62
Fphi8B=
;
	full_text.
,
*%63 = phi float [ %58, %61 ], [ %71, %62 ]
)float8B

	full_text

	float %58
)float8B

	full_text

	float %71
Dphi8B;
9
	full_text,
*
(%64 = phi i64 [ %59, %61 ], [ %72, %62 ]
%i648B

	full_text
	
i64 %59
%i648B

	full_text
	
i64 %72
Dphi8B;
9
	full_text,
*
(%65 = phi i64 [ %19, %61 ], [ %73, %62 ]
%i648B

	full_text
	
i64 %19
%i648B

	full_text
	
i64 %73
6add8B-
+
	full_text

%66 = add nsw i64 %64, %16
%i648B

	full_text
	
i64 %64
%i648B

	full_text
	
i64 %16
\getelementptr8BI
G
	full_text:
8
6%67 = getelementptr inbounds float, float* %0, i64 %66
%i648B

	full_text
	
i64 %66
Lload8BB
@
	full_text3
1
/%68 = load float, float* %67, align 4, !tbaa !9
+float*8B

	full_text


float* %67
\getelementptr8BI
G
	full_text:
8
6%69 = getelementptr inbounds float, float* %1, i64 %64
%i648B

	full_text
	
i64 %64
Lload8BB
@
	full_text3
1
/%70 = load float, float* %69, align 4, !tbaa !9
+float*8B

	full_text


float* %69
ecall8B[
Y
	full_textL
J
H%71 = tail call float @llvm.fmuladd.f32(float %68, float %70, float %63)
)float8B

	full_text

	float %68
)float8B

	full_text

	float %70
)float8B

	full_text

	float %63
Lstore8BA
?
	full_text2
0
.store float %71, float* %12, align 4, !tbaa !9
)float8B

	full_text

	float %71
+float*8B

	full_text


float* %12
8add8B/
-
	full_text 

%72 = add nuw nsw i64 %64, 1
%i648B

	full_text
	
i64 %64
1add8B(
&
	full_text

%73 = add i64 %65, -1
%i648B

	full_text
	
i64 %65
5icmp8B+
)
	full_text

%74 = icmp eq i64 %73, 0
%i648B

	full_text
	
i64 %73
Jbr8BB
@
	full_text3
1
/br i1 %74, label %75, label %62, !llvm.loop !13
#i18B

	full_text


i1 %74
$ret8B

	full_text


ret void
*float*8	B

	full_text

	float* %0
*float*8	B

	full_text

	float* %2
*float*8	B

	full_text

	float* %1
$i328	B

	full_text


i32 %3
$i328	B
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
$i648	B

	full_text


i64 32
2float8	B%
#
	full_text

float 0.000000e+00
#i648	B

	full_text	

i64 0
#i648	B

	full_text	

i64 1
#i648	B

	full_text	

i64 2
#i328	B

	full_text	

i32 0
#i648	B

	full_text	

i64 4
$i648	B

	full_text


i64 -1
#i648	B

	full_text	

i64 3
$i648	B

	full_text


i64 -4      	  
 

                   !  "    #% $$ &' && () (* (( +, +- ++ ./ .. 01 00 23 22 45 44 67 68 69 66 :; :< :: => == ?@ ?A ?? BC BB DE DD FG FF HI HH JK JL JM JJ NO NP NN QR QQ ST SU SS VW VV XY XX Z[ ZZ \] \\ ^_ ^` ^a ^^ bc bd bb ef ee gh gi gg jk jj lm ll no nn pq pp rs rt ru rr vw vx vv yz yy {| {{ }~ }} Ä 
Ç ÅÅ É
Ñ ÉÉ ÖÜ ÖÖ áà áã ä
å ää çé ç
è çç êë ê
í êê ìî ì
ï ìì ñ
ó ññ òô òò ö
õ öö úù úú ûü û
† û
° ûû ¢£ ¢
§ ¢¢ •¶ •• ß® ßß ©™ ©© ´¨ ´Æ .Æ BÆ VÆ jÆ ñØ ∞ 2∞ F∞ Z∞ n∞ ö	± ≤ 	≤ ≤     	 
          ! "r %y '  ){ *& , -+ /. 1& 32 50 74 8$ 96 ; <& >= @ A? CB E= GF ID KH L6 MJ O P& RQ T US WV YQ [Z ]X _\ `J a^ c d& fe h ig kj me on ql sp t^ ur w x& z( |{ ~} Är Çy Ñ ÜÖ àÅ ãû åÉ é• è ëß íç î ïì óñ ôç õö ùò üú †ä °û £ §ç ¶ê ®ß ™© ¨  ≠  ≠ Å  á ≠á â# $â ä Å $´ ≠´ ä ≥≥ ≠ ¥¥r ¥¥ rû ¥¥ û^ ¥¥ ^ ≥≥ J ¥¥ J6 ¥¥ 6	µ 	µ 
∂ ∂ $∂ Å∑ &	∑ }∑ É
∑ Ö
∑ ©	∏ =
∏ •	π Q∫ 	∫ 	ª y	º 
º ß	Ω 	Ω 	Ω e	æ {"
bicgKernel1"
_Z13get_global_idj"
llvm.fmuladd.f32*û
%polybench-gpu-1.0-bicg-bicgKernel1.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Å

devmap_label


wgsize_log1p
≥.êA

transfer_bytes
ÄÄÑ 
 
transfer_bytes_log1p
≥.êA

wgsize
Ä