
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
5icmpB-
+
	full_text

%11 = icmp slt i32 %10, %7
#i32B

	full_text
	
i32 %10
8brB2
0
	full_text#
!
br i1 %11, label %12, label %85
!i1B

	full_text


i1 %11
5icmp8B+
)
	full_text

%13 = icmp sgt i32 %7, 0
:br8B2
0
	full_text#
!
br i1 %13, label %19, label %14
#i18B

	full_text


i1 %13
0shl8B'
%
	full_text

%15 = shl i64 %9, 32
$i648B

	full_text


i64 %9
9ashr8B/
-
	full_text 

%16 = ashr exact i64 %15, 32
%i648B

	full_text
	
i64 %15
\getelementptr8BI
G
	full_text:
8
6%17 = getelementptr inbounds float, float* %4, i64 %16
%i648B

	full_text
	
i64 %16
\getelementptr8BI
G
	full_text:
8
6%18 = getelementptr inbounds float, float* %3, i64 %16
%i648B

	full_text
	
i64 %16
'br8B

	full_text

br label %78
5mul8B,
*
	full_text

%20 = mul nsw i32 %10, %7
%i328B

	full_text
	
i32 %10
0shl8B'
%
	full_text

%21 = shl i64 %9, 32
$i648B

	full_text


i64 %9
9ashr8B/
-
	full_text 

%22 = ashr exact i64 %21, 32
%i648B

	full_text
	
i64 %21
\getelementptr8BI
G
	full_text:
8
6%23 = getelementptr inbounds float, float* %4, i64 %22
%i648B

	full_text
	
i64 %22
\getelementptr8BI
G
	full_text:
8
6%24 = getelementptr inbounds float, float* %3, i64 %22
%i648B

	full_text
	
i64 %22
6sext8B,
*
	full_text

%25 = sext i32 %20 to i64
%i328B

	full_text
	
i32 %20
5zext8B+
)
	full_text

%26 = zext i32 %7 to i64
0and8B'
%
	full_text

%27 = and i64 %26, 1
%i648B

	full_text
	
i64 %26
4icmp8B*
(
	full_text

%28 = icmp eq i32 %7, 1
:br8B2
0
	full_text#
!
br i1 %28, label %62, label %29
#i18B

	full_text


i1 %28
6sub8B-
+
	full_text

%30 = sub nsw i64 %26, %27
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %27
'br8B

	full_text

br label %31
Bphi8B9
7
	full_text*
(
&%32 = phi i64 [ 0, %29 ], [ %59, %31 ]
%i648B

	full_text
	
i64 %59
Dphi8B;
9
	full_text,
*
(%33 = phi i64 [ %30, %29 ], [ %60, %31 ]
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %60
6add8B-
+
	full_text

%34 = add nsw i64 %32, %25
%i648B

	full_text
	
i64 %32
%i648B

	full_text
	
i64 %25
\getelementptr8BI
G
	full_text:
8
6%35 = getelementptr inbounds float, float* %0, i64 %34
%i648B

	full_text
	
i64 %34
Lload8BB
@
	full_text3
1
/%36 = load float, float* %35, align 4, !tbaa !9
+float*8B

	full_text


float* %35
\getelementptr8BI
G
	full_text:
8
6%37 = getelementptr inbounds float, float* %2, i64 %32
%i648B

	full_text
	
i64 %32
Lload8BB
@
	full_text3
1
/%38 = load float, float* %37, align 4, !tbaa !9
+float*8B

	full_text


float* %37
Lload8BB
@
	full_text3
1
/%39 = load float, float* %23, align 4, !tbaa !9
+float*8B

	full_text


float* %23
ecall8B[
Y
	full_textL
J
H%40 = tail call float @llvm.fmuladd.f32(float %36, float %38, float %39)
)float8B

	full_text

	float %36
)float8B

	full_text

	float %38
)float8B

	full_text

	float %39
Lstore8BA
?
	full_text2
0
.store float %40, float* %23, align 4, !tbaa !9
)float8B

	full_text

	float %40
+float*8B

	full_text


float* %23
\getelementptr8BI
G
	full_text:
8
6%41 = getelementptr inbounds float, float* %1, i64 %34
%i648B

	full_text
	
i64 %34
Lload8BB
@
	full_text3
1
/%42 = load float, float* %41, align 4, !tbaa !9
+float*8B

	full_text


float* %41
Lload8BB
@
	full_text3
1
/%43 = load float, float* %37, align 4, !tbaa !9
+float*8B

	full_text


float* %37
Lload8BB
@
	full_text3
1
/%44 = load float, float* %24, align 4, !tbaa !9
+float*8B

	full_text


float* %24
ecall8B[
Y
	full_textL
J
H%45 = tail call float @llvm.fmuladd.f32(float %42, float %43, float %44)
)float8B

	full_text

	float %42
)float8B

	full_text

	float %43
)float8B

	full_text

	float %44
Lstore8BA
?
	full_text2
0
.store float %45, float* %24, align 4, !tbaa !9
)float8B

	full_text

	float %45
+float*8B

	full_text


float* %24
.or8B&
$
	full_text

%46 = or i64 %32, 1
%i648B

	full_text
	
i64 %32
6add8B-
+
	full_text

%47 = add nsw i64 %46, %25
%i648B

	full_text
	
i64 %46
%i648B

	full_text
	
i64 %25
\getelementptr8BI
G
	full_text:
8
6%48 = getelementptr inbounds float, float* %0, i64 %47
%i648B

	full_text
	
i64 %47
Lload8BB
@
	full_text3
1
/%49 = load float, float* %48, align 4, !tbaa !9
+float*8B

	full_text


float* %48
\getelementptr8BI
G
	full_text:
8
6%50 = getelementptr inbounds float, float* %2, i64 %46
%i648B

	full_text
	
i64 %46
Lload8BB
@
	full_text3
1
/%51 = load float, float* %50, align 4, !tbaa !9
+float*8B

	full_text


float* %50
Lload8BB
@
	full_text3
1
/%52 = load float, float* %23, align 4, !tbaa !9
+float*8B

	full_text


float* %23
ecall8B[
Y
	full_textL
J
H%53 = tail call float @llvm.fmuladd.f32(float %49, float %51, float %52)
)float8B

	full_text

	float %49
)float8B

	full_text

	float %51
)float8B

	full_text

	float %52
Lstore8BA
?
	full_text2
0
.store float %53, float* %23, align 4, !tbaa !9
)float8B

	full_text

	float %53
+float*8B

	full_text


float* %23
\getelementptr8BI
G
	full_text:
8
6%54 = getelementptr inbounds float, float* %1, i64 %47
%i648B

	full_text
	
i64 %47
Lload8BB
@
	full_text3
1
/%55 = load float, float* %54, align 4, !tbaa !9
+float*8B

	full_text


float* %54
Lload8BB
@
	full_text3
1
/%56 = load float, float* %50, align 4, !tbaa !9
+float*8B

	full_text


float* %50
Lload8BB
@
	full_text3
1
/%57 = load float, float* %24, align 4, !tbaa !9
+float*8B

	full_text


float* %24
ecall8B[
Y
	full_textL
J
H%58 = tail call float @llvm.fmuladd.f32(float %55, float %56, float %57)
)float8B

	full_text

	float %55
)float8B

	full_text

	float %56
)float8B

	full_text

	float %57
Lstore8BA
?
	full_text2
0
.store float %58, float* %24, align 4, !tbaa !9
)float8B

	full_text

	float %58
+float*8B

	full_text


float* %24
4add8B+
)
	full_text

%59 = add nsw i64 %32, 2
%i648B

	full_text
	
i64 %32
1add8B(
&
	full_text

%60 = add i64 %33, -2
%i648B

	full_text
	
i64 %33
5icmp8B+
)
	full_text

%61 = icmp eq i64 %60, 0
%i648B

	full_text
	
i64 %60
:br8B2
0
	full_text#
!
br i1 %61, label %62, label %31
#i18B

	full_text


i1 %61
Bphi8B9
7
	full_text*
(
&%63 = phi i64 [ 0, %19 ], [ %59, %31 ]
%i648B

	full_text
	
i64 %59
5icmp8B+
)
	full_text

%64 = icmp eq i64 %27, 0
%i648B

	full_text
	
i64 %27
:br8B2
0
	full_text#
!
br i1 %64, label %78, label %65
#i18B

	full_text


i1 %64
6add8B-
+
	full_text

%66 = add nsw i64 %63, %25
%i648B

	full_text
	
i64 %63
%i648B

	full_text
	
i64 %25
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
6%69 = getelementptr inbounds float, float* %2, i64 %63
%i648B

	full_text
	
i64 %63
Lload8BB
@
	full_text3
1
/%70 = load float, float* %69, align 4, !tbaa !9
+float*8B

	full_text


float* %69
Lload8BB
@
	full_text3
1
/%71 = load float, float* %23, align 4, !tbaa !9
+float*8B

	full_text


float* %23
ecall8B[
Y
	full_textL
J
H%72 = tail call float @llvm.fmuladd.f32(float %68, float %70, float %71)
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

	float %71
Lstore8BA
?
	full_text2
0
.store float %72, float* %23, align 4, !tbaa !9
)float8B

	full_text

	float %72
+float*8B

	full_text


float* %23
\getelementptr8BI
G
	full_text:
8
6%73 = getelementptr inbounds float, float* %1, i64 %66
%i648B

	full_text
	
i64 %66
Lload8BB
@
	full_text3
1
/%74 = load float, float* %73, align 4, !tbaa !9
+float*8B

	full_text


float* %73
Lload8BB
@
	full_text3
1
/%75 = load float, float* %69, align 4, !tbaa !9
+float*8B

	full_text


float* %69
Lload8BB
@
	full_text3
1
/%76 = load float, float* %24, align 4, !tbaa !9
+float*8B

	full_text


float* %24
ecall8B[
Y
	full_textL
J
H%77 = tail call float @llvm.fmuladd.f32(float %74, float %75, float %76)
)float8B

	full_text

	float %74
)float8B

	full_text

	float %75
)float8B

	full_text

	float %76
Lstore8BA
?
	full_text2
0
.store float %77, float* %24, align 4, !tbaa !9
)float8B

	full_text

	float %77
+float*8B

	full_text


float* %24
'br8B

	full_text

br label %78
Uphi8BL
J
	full_text=
;
9%79 = phi float* [ %18, %14 ], [ %24, %62 ], [ %24, %65 ]
+float*8B

	full_text


float* %18
+float*8B

	full_text


float* %24
+float*8B

	full_text


float* %24
Uphi8BL
J
	full_text=
;
9%80 = phi float* [ %17, %14 ], [ %23, %62 ], [ %23, %65 ]
+float*8B

	full_text


float* %17
+float*8B

	full_text


float* %23
+float*8B

	full_text


float* %23
Lload8BB
@
	full_text3
1
/%81 = load float, float* %80, align 4, !tbaa !9
+float*8B

	full_text


float* %80
Lload8BB
@
	full_text3
1
/%82 = load float, float* %79, align 4, !tbaa !9
+float*8B

	full_text


float* %79
5fmul8B+
)
	full_text

%83 = fmul float %82, %6
)float8B

	full_text

	float %82
dcall8BZ
X
	full_textK
I
G%84 = tail call float @llvm.fmuladd.f32(float %5, float %81, float %83)
)float8B

	full_text

	float %81
)float8B

	full_text

	float %83
Lstore8BA
?
	full_text2
0
.store float %84, float* %79, align 4, !tbaa !9
)float8B

	full_text

	float %84
+float*8B

	full_text


float* %79
'br8B

	full_text

br label %85
$ret8	B

	full_text


ret void
*float*8
B

	full_text

	float* %4
$i328
B

	full_text


i32 %7
*float*8
B

	full_text

	float* %3
*float*8
B

	full_text

	float* %2
(float8
B

	full_text


float %5
*float*8
B

	full_text

	float* %0
*float*8
B

	full_text

	float* %1
(float8
B

	full_text


float %6
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
#i648
B

	full_text	

i64 2
#i328
B

	full_text	

i32 0
#i648
B

	full_text	

i64 0
$i648
B

	full_text


i64 -2
#i648
B

	full_text	

i64 1
$i648
B

	full_text


i64 32
#i328
B

	full_text	

i32 1       	
 	                       !" !! ## $% $' &( && )+ ** ,- ,. ,, /0 /1 // 23 22 45 44 67 66 89 88 :; :: <= <> <? << @A @B @@ CD CC EF EE GH GG IJ II KL KM KN KK OP OQ OO RS RR TU TV TT WX WW YZ YY [\ [[ ]^ ]] _` __ ab ac ad aa ef eg ee hi hh jk jj lm ll no nn pq pr ps pp tu tv tt wx ww yz yy {| {{ }~ }	€  ‚  ƒ„ ƒ† …
‡ …… ˆ
‰ ˆˆ Š‹ ŠŠ Œ
 ŒŒ Ž ŽŽ ‘  ’“ ’
” ’
• ’’ –— –
˜ –– ™
š ™™ ›œ ›› ž  Ÿ  ŸŸ ¡¢ ¡
£ ¡
¤ ¡¡ ¥¦ ¥
§ ¥¥ ¨ª ©
« ©
¬ ©© ­® ­
¯ ­
° ­­ ±² ±± ³´ ³³ µ¶ µµ ·
¸ ·
¹ ·· º» º
¼ ºº ½¿ ¿ 	À À 	À À  À #Á Á Â 6Â [Â ŒÃ ·Ä 2Ä WÄ ˆÅ CÅ hÅ ™
Æ µ    
            "# %  '! (w +& -y .* 0 1/ 32 5* 76 9 ;4 =8 >: ?< A B/ DC F6 H JE LG MI NK P Q* SR U VT XW ZR \[ ^ `Y b] c_ da f gT ih k[ m oj ql rn sp u v* x, zy |{ ~w €! ‚ „ † ‡… ‰ˆ ‹ Œ  ‘Š “Ž ” •’ — ˜… š™ œŒ ž  › ¢ £Ÿ ¤¡ ¦ § ª « ¬ ® ¯ °­ ²© ´³ ¶± ¸µ ¹· »© ¼  ¾	 	 $ $ & ©ƒ ©ƒ …) *½ ¾¨ ©} } * ¾ ÈÈ ÇÇ< ÈÈ <p ÈÈ p¡ ÈÈ ¡’ ÈÈ ’· ÈÈ ·a ÈÈ a ÇÇ K ÈÈ K	É wÊ 	Ê Ë *	Ë {Ë 
Ë 	Ì y	Í !	Í R	Î 	Î 	Î 	Î 	Ï #"
gesummv_kernel"
_Z13get_global_idj"
llvm.fmuladd.f32*¤
+polybench-gpu-1.0-gesummv-gesummv_kernel.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282

devmap_label
 

wgsize_log1p
¹•A

wgsize
€
 
transfer_bytes_log1p
¹•A

transfer_bytes
€€ƒ@