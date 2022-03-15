

[external]
KcallBC
A
	full_text4
2
0%7 = tail call i64 @_Z13get_global_idj(i32 0) #3
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
LextractelementB:
8
	full_text+
)
'%9 = extractelement <2 x i32> %3, i64 0
0uremB(
&
	full_text

%10 = urem i32 %8, %9
"i32B

	full_text


i32 %8
"i32B

	full_text


i32 %9
0udivB(
&
	full_text

%11 = udiv i32 %8, %9
"i32B

	full_text


i32 %8
"i32B

	full_text


i32 %9
MextractelementB;
9
	full_text,
*
(%12 = extractelement <2 x i32> %4, i64 0
MextractelementB;
9
	full_text,
*
(%13 = extractelement <2 x i32> %3, i64 1
6icmpB.
,
	full_text

%14 = icmp ult i32 %11, %13
#i32B

	full_text
	
i32 %11
#i32B

	full_text
	
i32 %13
9brB3
1
	full_text$
"
 br i1 %14, label %15, label %100
!i1B

	full_text


i1 %14
Oextractelement8B;
9
	full_text,
*
(%16 = extractelement <2 x i32> %4, i64 1
2add8B)
'
	full_text

%17 = add i32 %11, %16
%i328B

	full_text
	
i32 %11
%i328B

	full_text
	
i32 %16
8icmp8B.
,
	full_text

%18 = icmp ult i32 %11, %17
%i328B

	full_text
	
i32 %11
%i328B

	full_text
	
i32 %17
:br8B2
0
	full_text#
!
br i1 %18, label %19, label %28
#i18B

	full_text


i1 %18
2add8B)
'
	full_text

%20 = add i32 %10, %12
%i328B

	full_text
	
i32 %10
%i328B

	full_text
	
i32 %12
8icmp8B.
,
	full_text

%21 = icmp ult i32 %10, %20
%i328B

	full_text
	
i32 %10
%i328B

	full_text
	
i32 %20
6zext8B,
*
	full_text

%22 = zext i32 %10 to i64
%i328B

	full_text
	
i32 %10
6zext8B,
*
	full_text

%23 = zext i32 %12 to i64
%i328B

	full_text
	
i32 %12
0and8B'
%
	full_text

%24 = and i64 %23, 1
%i648B

	full_text
	
i64 %23
5icmp8B+
)
	full_text

%25 = icmp eq i32 %12, 1
%i328B

	full_text
	
i32 %12
6sub8B-
+
	full_text

%26 = sub nsw i64 %23, %24
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %24
5icmp8B+
)
	full_text

%27 = icmp eq i64 %24, 0
%i648B

	full_text
	
i64 %24
'br8B

	full_text

br label %34
Ophi8BF
D
	full_text7
5
3%29 = phi float [ 0.000000e+00, %15 ], [ %61, %60 ]
)float8B

	full_text

	float %61
?fadd8B5
3
	full_text&
$
"%30 = fadd float %29, 5.000000e-01
)float8B

	full_text

	float %29
<fptosi8B0
.
	full_text!

%31 = fptosi float %30 to i32
)float8B

	full_text

	float %30
8and8B/
-
	full_text 

%32 = and i64 %7, 4294967295
$i648B

	full_text


i64 %7
Xgetelementptr8BE
C
	full_text6
4
2%33 = getelementptr inbounds i32, i32* %2, i64 %32
%i648B

	full_text
	
i64 %32
Hstore8B=
;
	full_text.
,
*store i32 %31, i32* %33, align 4, !tbaa !9
%i328B

	full_text
	
i32 %31
'i32*8B

	full_text


i32* %33
(br8B 

	full_text

br label %100
Ophi8BF
D
	full_text7
5
3%35 = phi float [ 0.000000e+00, %19 ], [ %61, %60 ]
)float8B

	full_text

	float %61
Dphi8B;
9
	full_text,
*
(%36 = phi i32 [ %11, %19 ], [ %62, %60 ]
%i328B

	full_text
	
i32 %11
%i328B

	full_text
	
i32 %62
Bphi8B9
7
	full_text*
(
&%37 = phi i32 [ 0, %19 ], [ %63, %60 ]
%i328B

	full_text
	
i32 %63
:br8B2
0
	full_text#
!
br i1 %21, label %38, label %60
#i18B

	full_text


i1 %21
2mul8B)
'
	full_text

%39 = mul i32 %37, %12
%i328B

	full_text
	
i32 %37
%i328B

	full_text
	
i32 %12
1mul8B(
&
	full_text

%40 = mul i32 %36, %5
%i328B

	full_text
	
i32 %36
:br8B2
0
	full_text#
!
br i1 %25, label %42, label %41
#i18B

	full_text


i1 %25
'br8B

	full_text

br label %65
Hphi8B?
=
	full_text0
.
,%43 = phi float [ undef, %38 ], [ %95, %65 ]
)float8B

	full_text

	float %95
Bphi8B9
7
	full_text*
(
&%44 = phi i64 [ 0, %38 ], [ %97, %65 ]
%i648B

	full_text
	
i64 %97
Dphi8B;
9
	full_text,
*
(%45 = phi i64 [ %22, %38 ], [ %96, %65 ]
%i648B

	full_text
	
i64 %22
%i648B

	full_text
	
i64 %96
Fphi8B=
;
	full_text.
,
*%46 = phi float [ %35, %38 ], [ %95, %65 ]
)float8B

	full_text

	float %35
)float8B

	full_text

	float %95
:br8B2
0
	full_text#
!
br i1 %27, label %60, label %47
#i18B

	full_text


i1 %27
8trunc8B-
+
	full_text

%48 = trunc i64 %45 to i32
%i648B

	full_text
	
i64 %45
2add8B)
'
	full_text

%49 = add i32 %40, %48
%i328B

	full_text
	
i32 %40
%i328B

	full_text
	
i32 %48
6zext8B,
*
	full_text

%50 = zext i32 %49 to i64
%i328B

	full_text
	
i32 %49
Xgetelementptr8BE
C
	full_text6
4
2%51 = getelementptr inbounds i32, i32* %0, i64 %50
%i648B

	full_text
	
i64 %50
Hload8B>
<
	full_text/
-
+%52 = load i32, i32* %51, align 4, !tbaa !9
'i32*8B

	full_text


i32* %51
<uitofp8B0
.
	full_text!

%53 = uitofp i32 %52 to float
%i328B

	full_text
	
i32 %52
8trunc8B-
+
	full_text

%54 = trunc i64 %44 to i32
%i648B

	full_text
	
i64 %44
2add8B)
'
	full_text

%55 = add i32 %39, %54
%i328B

	full_text
	
i32 %39
%i328B

	full_text
	
i32 %54
6zext8B,
*
	full_text

%56 = zext i32 %55 to i64
%i328B

	full_text
	
i32 %55
\getelementptr8BI
G
	full_text:
8
6%57 = getelementptr inbounds float, float* %1, i64 %56
%i648B

	full_text
	
i64 %56
Mload8BC
A
	full_text4
2
0%58 = load float, float* %57, align 4, !tbaa !13
+float*8B

	full_text


float* %57
ecall8B[
Y
	full_textL
J
H%59 = tail call float @llvm.fmuladd.f32(float %53, float %58, float %46)
)float8B

	full_text

	float %53
)float8B

	full_text

	float %58
)float8B

	full_text

	float %46
'br8B

	full_text

br label %60
Tphi8	BK
I
	full_text<
:
8%61 = phi float [ %35, %34 ], [ %43, %42 ], [ %59, %47 ]
)float8	B

	full_text

	float %35
)float8	B

	full_text

	float %43
)float8	B

	full_text

	float %59
4add8	B+
)
	full_text

%62 = add nuw i32 %36, 1
%i328	B

	full_text
	
i32 %36
8add8	B/
-
	full_text 

%63 = add nuw nsw i32 %37, 1
%i328	B

	full_text
	
i32 %37
7icmp8	B-
+
	full_text

%64 = icmp eq i32 %63, %16
%i328	B

	full_text
	
i32 %63
%i328	B

	full_text
	
i32 %16
:br8	B2
0
	full_text#
!
br i1 %64, label %28, label %34
#i18	B

	full_text


i1 %64
Bphi8
B9
7
	full_text*
(
&%66 = phi i64 [ 0, %41 ], [ %97, %65 ]
%i648
B

	full_text
	
i64 %97
Dphi8
B;
9
	full_text,
*
(%67 = phi i64 [ %22, %41 ], [ %96, %65 ]
%i648
B

	full_text
	
i64 %22
%i648
B

	full_text
	
i64 %96
Fphi8
B=
;
	full_text.
,
*%68 = phi float [ %35, %41 ], [ %95, %65 ]
)float8
B

	full_text

	float %35
)float8
B

	full_text

	float %95
Dphi8
B;
9
	full_text,
*
(%69 = phi i64 [ %26, %41 ], [ %98, %65 ]
%i648
B

	full_text
	
i64 %26
%i648
B

	full_text
	
i64 %98
8trunc8
B-
+
	full_text

%70 = trunc i64 %66 to i32
%i648
B

	full_text
	
i64 %66
2add8
B)
'
	full_text

%71 = add i32 %39, %70
%i328
B

	full_text
	
i32 %39
%i328
B

	full_text
	
i32 %70
8trunc8
B-
+
	full_text

%72 = trunc i64 %67 to i32
%i648
B

	full_text
	
i64 %67
2add8
B)
'
	full_text

%73 = add i32 %40, %72
%i328
B

	full_text
	
i32 %40
%i328
B

	full_text
	
i32 %72
6zext8
B,
*
	full_text

%74 = zext i32 %73 to i64
%i328
B

	full_text
	
i32 %73
Xgetelementptr8
BE
C
	full_text6
4
2%75 = getelementptr inbounds i32, i32* %0, i64 %74
%i648
B

	full_text
	
i64 %74
Hload8
B>
<
	full_text/
-
+%76 = load i32, i32* %75, align 4, !tbaa !9
'i32*8
B

	full_text


i32* %75
<uitofp8
B0
.
	full_text!

%77 = uitofp i32 %76 to float
%i328
B

	full_text
	
i32 %76
6zext8
B,
*
	full_text

%78 = zext i32 %71 to i64
%i328
B

	full_text
	
i32 %71
\getelementptr8
BI
G
	full_text:
8
6%79 = getelementptr inbounds float, float* %1, i64 %78
%i648
B

	full_text
	
i64 %78
Mload8
BC
A
	full_text4
2
0%80 = load float, float* %79, align 4, !tbaa !13
+float*8
B

	full_text


float* %79
ecall8
B[
Y
	full_textL
J
H%81 = tail call float @llvm.fmuladd.f32(float %77, float %80, float %68)
)float8
B

	full_text

	float %77
)float8
B

	full_text

	float %80
)float8
B

	full_text

	float %68
8add8
B/
-
	full_text 

%82 = add nuw nsw i64 %67, 1
%i648
B

	full_text
	
i64 %67
8trunc8
B-
+
	full_text

%83 = trunc i64 %66 to i32
%i648
B

	full_text
	
i64 %66
.or8
B&
$
	full_text

%84 = or i32 %83, 1
%i328
B

	full_text
	
i32 %83
2add8
B)
'
	full_text

%85 = add i32 %39, %84
%i328
B

	full_text
	
i32 %39
%i328
B

	full_text
	
i32 %84
8trunc8
B-
+
	full_text

%86 = trunc i64 %82 to i32
%i648
B

	full_text
	
i64 %82
2add8
B)
'
	full_text

%87 = add i32 %40, %86
%i328
B

	full_text
	
i32 %40
%i328
B

	full_text
	
i32 %86
6zext8
B,
*
	full_text

%88 = zext i32 %87 to i64
%i328
B

	full_text
	
i32 %87
Xgetelementptr8
BE
C
	full_text6
4
2%89 = getelementptr inbounds i32, i32* %0, i64 %88
%i648
B

	full_text
	
i64 %88
Hload8
B>
<
	full_text/
-
+%90 = load i32, i32* %89, align 4, !tbaa !9
'i32*8
B

	full_text


i32* %89
<uitofp8
B0
.
	full_text!

%91 = uitofp i32 %90 to float
%i328
B

	full_text
	
i32 %90
6zext8
B,
*
	full_text

%92 = zext i32 %85 to i64
%i328
B

	full_text
	
i32 %85
\getelementptr8
BI
G
	full_text:
8
6%93 = getelementptr inbounds float, float* %1, i64 %92
%i648
B

	full_text
	
i64 %92
Mload8
BC
A
	full_text4
2
0%94 = load float, float* %93, align 4, !tbaa !13
+float*8
B

	full_text


float* %93
ecall8
B[
Y
	full_textL
J
H%95 = tail call float @llvm.fmuladd.f32(float %91, float %94, float %81)
)float8
B

	full_text

	float %91
)float8
B

	full_text

	float %94
)float8
B

	full_text

	float %81
4add8
B+
)
	full_text

%96 = add nsw i64 %67, 2
%i648
B

	full_text
	
i64 %67
4add8
B+
)
	full_text

%97 = add nsw i64 %66, 2
%i648
B

	full_text
	
i64 %66
1add8
B(
&
	full_text

%98 = add i64 %69, -2
%i648
B

	full_text
	
i64 %69
5icmp8
B+
)
	full_text

%99 = icmp eq i64 %98, 0
%i648
B

	full_text
	
i64 %98
:br8
B2
0
	full_text#
!
br i1 %99, label %42, label %65
#i18
B

	full_text


i1 %99
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %5
&i32*8B

	full_text
	
i32* %0
*float*8B

	full_text

	float* %1
0	<2 x i32>8B

	full_text

<2 x i32> %4
0	<2 x i32>8B

	full_text

<2 x i32> %3
&i32*8B

	full_text
	
i32* %2
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
,i648B!

	full_text

i64 4294967295
2float8B%
#
	full_text

float 0.000000e+00
2float8B%
#
	full_text

float 5.000000e-01
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 1
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 2
+float8B

	full_text

float undef
#i328B

	full_text	

i32 1
$i648B

	full_text


i64 -2       	 
                       !" !! #$ ## %& %% '( '' )* )+ )) ,- ,, .0 // 12 11 34 33 56 55 78 77 9: 9; 99 <> == ?@ ?A ?? BC BB DE DG FH FF IJ II KL KO NN PQ PP RS RT RR UV UW UU XY X[ ZZ \] \^ \\ _` __ ab aa cd cc ef ee gh gg ij ik ii lm ll no nn pq pp rs rt ru rr vx wy wz ww {| {{ }~ }} Ä 	Å  ÇÉ Ç
Ö ÑÑ Üá Ü
à ÜÜ âä â
ã ââ åç å
é åå èê èè ëí ë
ì ëë îï îî ñó ñ
ò ññ ôö ôô õ
ú õõ ùû ùù ü† üü °¢ °° £
§ ££ •¶ •• ß® ß
© ß
™ ßß ´¨ ´´ ≠Æ ≠≠ Ø∞ ØØ ±≤ ±
≥ ±± ¥µ ¥¥ ∂∑ ∂
∏ ∂∂ π∫ ππ ª
º ªª Ωæ ΩΩ ø¿ øø ¡¬ ¡¡ √
ƒ √√ ≈∆ ≈≈ «» «
… «
  «« ÀÃ ÀÀ ÕŒ ÕÕ œ– œœ —“ —— ”‘ ”	÷ I◊ a◊ õ◊ ªÿ nÿ £ÿ √Ÿ Ÿ ⁄ ⁄ € 7    	 
              " $# & (# *% +% -w 0/ 21 4 65 83 :7 ;w > @{ A} C EB G H? J' L« OÕ Q! SÀ T= V« W, YR [I ]Z ^\ `_ ba dc fP hF jg ki ml on qe sp tU u= xN yr z? |B ~} Ä Å ÉÕ Ö! áÀ à= ä« ã) çœ éÑ êF íè ìÜ ïI óî òñ öô úõ ûù †ë ¢° §£ ¶ü ®• ©â ™Ü ¨Ñ Æ≠ ∞F ≤Ø ≥´ µI ∑¥ ∏∂ ∫π ºª æΩ ¿± ¬¡ ƒ√ ∆ø »≈ …ß  Ü ÃÑ Œå –œ “— ‘  ’  /. =< ’D FD wK NK MÇ /Ç =X wX ZM Ñv w” N” Ñ ’ ‹‹ ››« ›› «r ›› rß ›› ß ‹‹ 	ﬁ 5ﬂ /ﬂ =	‡ 1	· 	· 	· ,· P· Ñ
· —	‚ 	‚ 	‚ %
‚ ´„ „ B
‰ À
‰ ÕÂ N	Ê '	Ê {	Ê }
Ê Ø
Á œ"
simpleNonSeparableConvolution"
_Z13get_global_idj"
llvm.fmuladd.f32*´
2SimpleConvolution-simpleNonSeparableConvolution.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Å

transfer_bytes
Ã‡Ä

devmap_label


wgsize_log1p
tA
 
transfer_bytes_log1p
tA

wgsize
Ä