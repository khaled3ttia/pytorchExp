

[external]
KcallBC
A
	full_text4
2
0%9 = tail call i64 @_Z13get_global_idj(i32 0) #3
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
LcallBD
B
	full_text5
3
1%11 = tail call i64 @_Z13get_global_idj(i32 1) #3
6truncB-
+
	full_text

%12 = trunc i64 %11 to i32
#i64B

	full_text
	
i64 %11
5icmpB-
+
	full_text

%13 = icmp slt i32 %12, %5
#i32B

	full_text
	
i32 %12
5icmpB-
+
	full_text

%14 = icmp slt i32 %10, %6
#i32B

	full_text
	
i32 %10
/andB(
&
	full_text

%15 = and i1 %14, %13
!i1B

	full_text


i1 %14
!i1B

	full_text


i1 %13
8brB2
0
	full_text#
!
br i1 %15, label %16, label %75
!i1B

	full_text


i1 %15
5mul8B,
*
	full_text

%17 = mul nsw i32 %12, %6
%i328B

	full_text
	
i32 %12
6add8B-
+
	full_text

%18 = add nsw i32 %17, %10
%i328B

	full_text
	
i32 %17
%i328B

	full_text
	
i32 %10
6sext8B,
*
	full_text

%19 = sext i32 %18 to i64
%i328B

	full_text
	
i32 %18
\getelementptr8BI
G
	full_text:
8
6%20 = getelementptr inbounds float, float* %2, i64 %19
%i648B

	full_text
	
i64 %19
Lload8BB
@
	full_text3
1
/%21 = load float, float* %20, align 4, !tbaa !9
+float*8B

	full_text


float* %20
5fmul8B+
)
	full_text

%22 = fmul float %21, %4
)float8B

	full_text

	float %21
Lstore8BA
?
	full_text2
0
.store float %22, float* %20, align 4, !tbaa !9
)float8B

	full_text

	float %22
+float*8B

	full_text


float* %20
5icmp8B+
)
	full_text

%23 = icmp sgt i32 %7, 0
:br8B2
0
	full_text#
!
br i1 %23, label %24, label %75
#i18B

	full_text


i1 %23
5mul8B,
*
	full_text

%25 = mul nsw i32 %12, %7
%i328B

	full_text
	
i32 %12
5sext8B+
)
	full_text

%26 = sext i32 %6 to i64
0shl8B'
%
	full_text

%27 = shl i64 %9, 32
$i648B

	full_text


i64 %9
9ashr8B/
-
	full_text 

%28 = ashr exact i64 %27, 32
%i648B

	full_text
	
i64 %27
6sext8B,
*
	full_text

%29 = sext i32 %25 to i64
%i328B

	full_text
	
i32 %25
5zext8B+
)
	full_text

%30 = zext i32 %7 to i64
0and8B'
%
	full_text

%31 = and i64 %30, 1
%i648B

	full_text
	
i64 %30
4icmp8B*
(
	full_text

%32 = icmp eq i32 %7, 1
:br8B2
0
	full_text#
!
br i1 %32, label %61, label %33
#i18B

	full_text


i1 %32
6sub8B-
+
	full_text

%34 = sub nsw i64 %30, %31
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %31
'br8B

	full_text

br label %35
Fphi8B=
;
	full_text.
,
*%36 = phi float [ %22, %33 ], [ %57, %35 ]
)float8B

	full_text

	float %22
)float8B

	full_text

	float %57
Bphi8B9
7
	full_text*
(
&%37 = phi i64 [ 0, %33 ], [ %58, %35 ]
%i648B

	full_text
	
i64 %58
Dphi8B;
9
	full_text,
*
(%38 = phi i64 [ %34, %33 ], [ %59, %35 ]
%i648B

	full_text
	
i64 %34
%i648B

	full_text
	
i64 %59
6add8B-
+
	full_text

%39 = add nsw i64 %37, %29
%i648B

	full_text
	
i64 %37
%i648B

	full_text
	
i64 %29
\getelementptr8BI
G
	full_text:
8
6%40 = getelementptr inbounds float, float* %0, i64 %39
%i648B

	full_text
	
i64 %39
Lload8BB
@
	full_text3
1
/%41 = load float, float* %40, align 4, !tbaa !9
+float*8B

	full_text


float* %40
5fmul8B+
)
	full_text

%42 = fmul float %41, %3
)float8B

	full_text

	float %41
6mul8B-
+
	full_text

%43 = mul nsw i64 %37, %26
%i648B

	full_text
	
i64 %37
%i648B

	full_text
	
i64 %26
6add8B-
+
	full_text

%44 = add nsw i64 %43, %28
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %28
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
H%47 = tail call float @llvm.fmuladd.f32(float %42, float %46, float %36)
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

	float %36
Lstore8BA
?
	full_text2
0
.store float %47, float* %20, align 4, !tbaa !9
)float8B

	full_text

	float %47
+float*8B

	full_text


float* %20
.or8B&
$
	full_text

%48 = or i64 %37, 1
%i648B

	full_text
	
i64 %37
6add8B-
+
	full_text

%49 = add nsw i64 %48, %29
%i648B

	full_text
	
i64 %48
%i648B

	full_text
	
i64 %29
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
5fmul8B+
)
	full_text

%52 = fmul float %51, %3
)float8B

	full_text

	float %51
6mul8B-
+
	full_text

%53 = mul nsw i64 %48, %26
%i648B

	full_text
	
i64 %48
%i648B

	full_text
	
i64 %26
6add8B-
+
	full_text

%54 = add nsw i64 %53, %28
%i648B

	full_text
	
i64 %53
%i648B

	full_text
	
i64 %28
\getelementptr8BI
G
	full_text:
8
6%55 = getelementptr inbounds float, float* %1, i64 %54
%i648B

	full_text
	
i64 %54
Lload8BB
@
	full_text3
1
/%56 = load float, float* %55, align 4, !tbaa !9
+float*8B

	full_text


float* %55
ecall8B[
Y
	full_textL
J
H%57 = tail call float @llvm.fmuladd.f32(float %52, float %56, float %47)
)float8B

	full_text

	float %52
)float8B

	full_text

	float %56
)float8B

	full_text

	float %47
Lstore8BA
?
	full_text2
0
.store float %57, float* %20, align 4, !tbaa !9
)float8B

	full_text

	float %57
+float*8B

	full_text


float* %20
4add8B+
)
	full_text

%58 = add nsw i64 %37, 2
%i648B

	full_text
	
i64 %37
1add8B(
&
	full_text

%59 = add i64 %38, -2
%i648B

	full_text
	
i64 %38
5icmp8B+
)
	full_text

%60 = icmp eq i64 %59, 0
%i648B

	full_text
	
i64 %59
:br8B2
0
	full_text#
!
br i1 %60, label %61, label %35
#i18B

	full_text


i1 %60
Fphi8B=
;
	full_text.
,
*%62 = phi float [ %22, %24 ], [ %57, %35 ]
)float8B

	full_text

	float %22
)float8B

	full_text

	float %57
Bphi8B9
7
	full_text*
(
&%63 = phi i64 [ 0, %24 ], [ %58, %35 ]
%i648B

	full_text
	
i64 %58
5icmp8B+
)
	full_text

%64 = icmp eq i64 %31, 0
%i648B

	full_text
	
i64 %31
:br8B2
0
	full_text#
!
br i1 %64, label %75, label %65
#i18B

	full_text


i1 %64
6add8B-
+
	full_text

%66 = add nsw i64 %63, %29
%i648B

	full_text
	
i64 %63
%i648B

	full_text
	
i64 %29
\getelementptr8BI
G
	full_text:
8
6%67 = getelementptr inbounds float, float* %0, i64 %66
%i648B

	full_text
	
i64 %66
Lload8BB
@
	full_text3
1
/%68 = load float, float* %67, align 4, !tbaa !9
+float*8B

	full_text


float* %67
5fmul8B+
)
	full_text

%69 = fmul float %68, %3
)float8B

	full_text

	float %68
6mul8B-
+
	full_text

%70 = mul nsw i64 %63, %26
%i648B

	full_text
	
i64 %63
%i648B

	full_text
	
i64 %26
6add8B-
+
	full_text

%71 = add nsw i64 %70, %28
%i648B

	full_text
	
i64 %70
%i648B

	full_text
	
i64 %28
\getelementptr8BI
G
	full_text:
8
6%72 = getelementptr inbounds float, float* %1, i64 %71
%i648B

	full_text
	
i64 %71
Lload8BB
@
	full_text3
1
/%73 = load float, float* %72, align 4, !tbaa !9
+float*8B

	full_text


float* %72
ecall8B[
Y
	full_textL
J
H%74 = tail call float @llvm.fmuladd.f32(float %69, float %73, float %62)
)float8B

	full_text

	float %69
)float8B

	full_text

	float %73
)float8B

	full_text

	float %62
Lstore8BA
?
	full_text2
0
.store float %74, float* %20, align 4, !tbaa !9
)float8B

	full_text

	float %74
+float*8B

	full_text


float* %20
'br8B

	full_text

br label %75
$ret8B

	full_text


ret void
*float*8B

	full_text

	float* %2
(float8B

	full_text


float %4
*float*8B

	full_text

	float* %0
$i328B

	full_text


i32 %5
$i328B

	full_text


i32 %7
$i328B

	full_text


i32 %6
*float*8B

	full_text

	float* %1
(float8B

	full_text


float %3
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
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 2
$i648B

	full_text


i64 -2
#i328B

	full_text	

i32 0
$i648B

	full_text


i64 32        	
 		                        !" !$ ## %% &' && () (( *+ ** ,, -. -- // 01 03 24 22 57 68 66 9: 99 ;< ;= ;; >? >@ >> AB AA CD CC EF EE GH GI GG JK JL JJ MN MM OP OO QR QS QT QQ UV UW UU XY XX Z[ Z\ ZZ ]^ ]] _` __ ab aa cd ce cc fg fh ff ij ii kl kk mn mo mp mm qr qs qq tu tt vw vv xy xx z{ z} |~ || 	Ä  ÅÇ ÅÅ ÉÑ ÉÜ Ö
á ÖÖ à
â àà äã ää åç åå éè é
ê éé ëí ë
ì ëë î
ï îî ñó ññ òô ò
ö ò
õ òò úù ú
û úú ü° 	¢ £ A£ ]£ à	§ •  	• #• ,• /	¶ 		¶ ¶ %ß Mß iß î	® E	® a
® å    
	              " $ '& )# +, ./ 1, 3- 4 7m 8t :2 <v =9 ?* @> BA DC F9 H% IG K( LJ NM PE RO S6 TQ V W9 YX [* \Z ^] `_ bX d% ec g( hf ji la nk oQ pm r s9 u; wv yx { }m ~t Ä- ÇÅ Ñ Ü* áÖ âà ãä ç è% êé í( ìë ïî óå ôñ ö| õò ù û  †! #! †0 |0 2É †É Ö5 6ü †z |z 6 ™™ † ©©m ™™ m ©© Q ™™ Q ©© ò ™™ ò´ 	´ /	¨ -	¨ X≠ 9	≠ x≠ 
≠ Å	Æ t	Ø v∞ 	∞  	± &	± ("
gemm"
_Z13get_global_idj"
llvm.fmuladd.f32*ó
polybench-gpu-1.0-gemm-gemm.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Å

devmap_label


transfer_bytes
ÄÄ¿

wgsize
Ä

wgsize_log1p
âboA
 
transfer_bytes_log1p
âboA