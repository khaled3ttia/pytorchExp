

[external]
LcallBD
B
	full_text5
3
1%10 = tail call i64 @_Z13get_global_idj(i32 0) #3
6truncB-
+
	full_text

%11 = trunc i64 %10 to i32
#i64B

	full_text
	
i64 %10
LcallBD
B
	full_text5
3
1%12 = tail call i64 @_Z13get_global_idj(i32 1) #3
6truncB-
+
	full_text

%13 = trunc i64 %12 to i32
#i64B

	full_text
	
i64 %12
5icmpB-
+
	full_text

%14 = icmp slt i32 %13, %3
#i32B

	full_text
	
i32 %13
5icmpB-
+
	full_text

%15 = icmp slt i32 %11, %6
#i32B

	full_text
	
i32 %11
/andB(
&
	full_text

%16 = and i1 %15, %14
!i1B

	full_text


i1 %15
!i1B

	full_text


i1 %14
8brB2
0
	full_text#
!
br i1 %16, label %17, label %73
!i1B

	full_text


i1 %16
5mul8B,
*
	full_text

%18 = mul nsw i32 %13, %6
%i328B

	full_text
	
i32 %13
6add8B-
+
	full_text

%19 = add nsw i32 %18, %11
%i328B

	full_text
	
i32 %18
%i328B

	full_text
	
i32 %11
6sext8B,
*
	full_text

%20 = sext i32 %19 to i64
%i328B

	full_text
	
i32 %19
\getelementptr8BI
G
	full_text:
8
6%21 = getelementptr inbounds float, float* %2, i64 %20
%i648B

	full_text
	
i64 %20
Lload8BB
@
	full_text3
1
/%22 = load float, float* %21, align 4, !tbaa !9
+float*8B

	full_text


float* %21
5fmul8B+
)
	full_text

%23 = fmul float %22, %8
)float8B

	full_text

	float %22
Lstore8BA
?
	full_text2
0
.store float %23, float* %21, align 4, !tbaa !9
)float8B

	full_text

	float %23
+float*8B

	full_text


float* %21
5icmp8B+
)
	full_text

%24 = icmp sgt i32 %4, 0
:br8B2
0
	full_text#
!
br i1 %24, label %25, label %73
#i18B

	full_text


i1 %24
5mul8B,
*
	full_text

%26 = mul nsw i32 %13, %4
%i328B

	full_text
	
i32 %13
5sext8B+
)
	full_text

%27 = sext i32 %6 to i64
1shl8B(
&
	full_text

%28 = shl i64 %10, 32
%i648B

	full_text
	
i64 %10
9ashr8B/
-
	full_text 

%29 = ashr exact i64 %28, 32
%i648B

	full_text
	
i64 %28
6sext8B,
*
	full_text

%30 = sext i32 %26 to i64
%i328B

	full_text
	
i32 %26
5zext8B+
)
	full_text

%31 = zext i32 %4 to i64
0and8B'
%
	full_text

%32 = and i64 %31, 1
%i648B

	full_text
	
i64 %31
4icmp8B*
(
	full_text

%33 = icmp eq i32 %4, 1
:br8B2
0
	full_text#
!
br i1 %33, label %60, label %34
#i18B

	full_text


i1 %33
6sub8B-
+
	full_text

%35 = sub nsw i64 %31, %32
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %32
'br8B

	full_text

br label %36
Fphi8B=
;
	full_text.
,
*%37 = phi float [ %23, %34 ], [ %56, %36 ]
)float8B

	full_text

	float %23
)float8B

	full_text

	float %56
Bphi8B9
7
	full_text*
(
&%38 = phi i64 [ 0, %34 ], [ %57, %36 ]
%i648B

	full_text
	
i64 %57
Dphi8B;
9
	full_text,
*
(%39 = phi i64 [ %35, %34 ], [ %58, %36 ]
%i648B

	full_text
	
i64 %35
%i648B

	full_text
	
i64 %58
6add8B-
+
	full_text

%40 = add nsw i64 %38, %30
%i648B

	full_text
	
i64 %38
%i648B

	full_text
	
i64 %30
\getelementptr8BI
G
	full_text:
8
6%41 = getelementptr inbounds float, float* %0, i64 %40
%i648B

	full_text
	
i64 %40
Lload8BB
@
	full_text3
1
/%42 = load float, float* %41, align 4, !tbaa !9
+float*8B

	full_text


float* %41
6mul8B-
+
	full_text

%43 = mul nsw i64 %38, %27
%i648B

	full_text
	
i64 %38
%i648B

	full_text
	
i64 %27
6add8B-
+
	full_text

%44 = add nsw i64 %43, %29
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %29
\getelementptr8BI
G
	full_text:
8
6%45 = getelementptr inbounds float, float* %1, i64 %44
%i648B

	full_text
	
i64 %44
Lload8BB
@
	full_text3
1
/%46 = load float, float* %45, align 4, !tbaa !9
+float*8B

	full_text


float* %45
ecall8B[
Y
	full_textL
J
H%47 = tail call float @llvm.fmuladd.f32(float %42, float %46, float %37)
)float8B

	full_text

	float %42
)float8B

	full_text

	float %46
)float8B

	full_text

	float %37
Lstore8BA
?
	full_text2
0
.store float %47, float* %21, align 4, !tbaa !9
)float8B

	full_text

	float %47
+float*8B

	full_text


float* %21
.or8B&
$
	full_text

%48 = or i64 %38, 1
%i648B

	full_text
	
i64 %38
6add8B-
+
	full_text

%49 = add nsw i64 %48, %30
%i648B

	full_text
	
i64 %48
%i648B

	full_text
	
i64 %30
\getelementptr8BI
G
	full_text:
8
6%50 = getelementptr inbounds float, float* %0, i64 %49
%i648B

	full_text
	
i64 %49
Lload8BB
@
	full_text3
1
/%51 = load float, float* %50, align 4, !tbaa !9
+float*8B

	full_text


float* %50
6mul8B-
+
	full_text

%52 = mul nsw i64 %48, %27
%i648B

	full_text
	
i64 %48
%i648B

	full_text
	
i64 %27
6add8B-
+
	full_text

%53 = add nsw i64 %52, %29
%i648B

	full_text
	
i64 %52
%i648B

	full_text
	
i64 %29
\getelementptr8BI
G
	full_text:
8
6%54 = getelementptr inbounds float, float* %1, i64 %53
%i648B

	full_text
	
i64 %53
Lload8BB
@
	full_text3
1
/%55 = load float, float* %54, align 4, !tbaa !9
+float*8B

	full_text


float* %54
ecall8B[
Y
	full_textL
J
H%56 = tail call float @llvm.fmuladd.f32(float %51, float %55, float %47)
)float8B

	full_text

	float %51
)float8B

	full_text

	float %55
)float8B

	full_text

	float %47
Lstore8BA
?
	full_text2
0
.store float %56, float* %21, align 4, !tbaa !9
)float8B

	full_text

	float %56
+float*8B

	full_text


float* %21
4add8B+
)
	full_text

%57 = add nsw i64 %38, 2
%i648B

	full_text
	
i64 %38
1add8B(
&
	full_text

%58 = add i64 %39, -2
%i648B

	full_text
	
i64 %39
5icmp8B+
)
	full_text

%59 = icmp eq i64 %58, 0
%i648B

	full_text
	
i64 %58
:br8B2
0
	full_text#
!
br i1 %59, label %60, label %36
#i18B

	full_text


i1 %59
Fphi8B=
;
	full_text.
,
*%61 = phi float [ %23, %25 ], [ %56, %36 ]
)float8B

	full_text

	float %23
)float8B

	full_text

	float %56
Bphi8B9
7
	full_text*
(
&%62 = phi i64 [ 0, %25 ], [ %57, %36 ]
%i648B

	full_text
	
i64 %57
5icmp8B+
)
	full_text

%63 = icmp eq i64 %32, 0
%i648B

	full_text
	
i64 %32
:br8B2
0
	full_text#
!
br i1 %63, label %73, label %64
#i18B

	full_text


i1 %63
6add8B-
+
	full_text

%65 = add nsw i64 %62, %30
%i648B

	full_text
	
i64 %62
%i648B

	full_text
	
i64 %30
\getelementptr8BI
G
	full_text:
8
6%66 = getelementptr inbounds float, float* %0, i64 %65
%i648B

	full_text
	
i64 %65
Lload8BB
@
	full_text3
1
/%67 = load float, float* %66, align 4, !tbaa !9
+float*8B

	full_text


float* %66
6mul8B-
+
	full_text

%68 = mul nsw i64 %62, %27
%i648B

	full_text
	
i64 %62
%i648B

	full_text
	
i64 %27
6add8B-
+
	full_text

%69 = add nsw i64 %68, %29
%i648B

	full_text
	
i64 %68
%i648B

	full_text
	
i64 %29
\getelementptr8BI
G
	full_text:
8
6%70 = getelementptr inbounds float, float* %1, i64 %69
%i648B

	full_text
	
i64 %69
Lload8BB
@
	full_text3
1
/%71 = load float, float* %70, align 4, !tbaa !9
+float*8B

	full_text


float* %70
ecall8B[
Y
	full_textL
J
H%72 = tail call float @llvm.fmuladd.f32(float %67, float %71, float %61)
)float8B

	full_text

	float %67
)float8B

	full_text

	float %71
)float8B

	full_text

	float %61
Lstore8BA
?
	full_text2
0
.store float %72, float* %21, align 4, !tbaa !9
)float8B

	full_text

	float %72
+float*8B

	full_text


float* %21
'br8B

	full_text

br label %73
$ret8B

	full_text


ret void
*float*8B

	full_text

	float* %1
(float8B

	full_text


float %8
*float*8B

	full_text

	float* %2
$i328B

	full_text


i32 %4
$i328B

	full_text


i32 %3
$i328B

	full_text


i32 %6
*float*8B
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
$i648B

	full_text


i64 32
#i328B

	full_text	

i32 0
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 2
#i648B

	full_text	

i64 1
$i648B

	full_text


i64 -2
#i648B

	full_text	

i64 0        	
 		                        !" !$ ## %% &' && () (( *+ ** ,, -. -- // 01 03 24 22 57 68 66 9: 99 ;< ;= ;; >? >@ >> AB AA CD CC EF EG EE HI HJ HH KL KK MN MM OP OQ OR OO ST SU SS VW VV XY XZ XX [\ [[ ]^ ]] _` _a __ bc bd bb ef ee gh gg ij ik il ii mn mo mm pq pp rs rr tu tt vw vy xz xx {| {{ }~ }} Ä Ç Å
É ÅÅ Ñ
Ö ÑÑ Üá ÜÜ àâ à
ä àà ãå ã
ç ãã é
è éé êë êê íì í
î í
ï íí ñó ñ
ò ññ ôõ Kõ eõ é	ú ù û  	û #û ,û /	ü 	† 		† † %° A° [° Ñ    
	              " $ '& )# +, ./ 1, 3- 4 7i 8p :2 <r =9 ?* @> BA D9 F% GE I( JH LK NC PM Q6 RO T U9 WV Y* ZX \[ ^V `% a_ c( db fe h] jg kO li n o9 q; sr ut w yi zp |- ~} Ä{ Ç* ÉÅ ÖÑ á{ â% äà å( çã èé ëÜ ìê îx ïí ó ò  ö! #! ö0 x0 2 ö Å5 6ô öv xv 6 ££ ö ¢¢O ££ Oi ££ ií ££ í ¢¢  ¢¢ 	§ &	§ (• 	•  ¶ 	¶ /	ß p	® -	® V	© r™ 9	™ t™ {	™ }"
mm2_kernel2"
_Z13get_global_idj"
llvm.fmuladd.f32*ù
$polybench-gpu-1.0-2mm-mm2_kernel2.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Å

transfer_bytes
ÄÄÄ(

devmap_label


wgsize
Ä

wgsize_log1p
≥ıëA
 
transfer_bytes_log1p
≥ıëA