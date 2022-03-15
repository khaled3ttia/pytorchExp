

[external]
JcallBB
@
	full_text3
1
/%9 = tail call i64 @_Z12get_group_idj(i32 1) #3
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
KcallBC
A
	full_text4
2
0%11 = tail call i64 @_Z12get_local_idj(i32 0) #3
6truncB-
+
	full_text

%12 = trunc i64 %11 to i32
#i64B

	full_text
	
i64 %11
KcallBC
A
	full_text4
2
0%13 = tail call i64 @_Z12get_local_idj(i32 1) #3
6truncB-
+
	full_text

%14 = trunc i64 %13 to i32
#i64B

	full_text
	
i64 %13
1addB*
(
	full_text

%15 = add nsw i32 %7, 1
.shlB'
%
	full_text

%16 = shl i32 %10, 4
#i32B

	full_text
	
i32 %10
0addB)
'
	full_text

%17 = add i32 %16, %14
#i32B

	full_text
	
i32 %16
#i32B

	full_text
	
i32 %14
0mulB)
'
	full_text

%18 = mul i32 %17, %15
#i32B

	full_text
	
i32 %17
#i32B

	full_text
	
i32 %15
-addB&
$
	full_text

%19 = add i32 %7, 2
0addB)
'
	full_text

%20 = add i32 %19, %12
#i32B

	full_text
	
i32 %19
#i32B

	full_text
	
i32 %12
0addB)
'
	full_text

%21 = add i32 %20, %18
#i32B

	full_text
	
i32 %20
#i32B

	full_text
	
i32 %18
3icmpB+
)
	full_text

%22 = icmp eq i32 %12, 0
#i32B

	full_text
	
i32 %12
8brB2
0
	full_text#
!
br i1 %22, label %27, label %23
!i1B

	full_text


i1 %22
1shl8B(
&
	full_text

%24 = shl i64 %13, 32
%i648B

	full_text
	
i64 %13
9ashr8B/
-
	full_text 

%25 = ashr exact i64 %24, 32
%i648B

	full_text
	
i64 %24
\getelementptr8BI
G
	full_text:
8
6%26 = getelementptr inbounds float, float* %4, i64 %25
%i648B

	full_text
	
i64 %25
'br8B

	full_text

br label %38
.or8B&
$
	full_text

%28 = or i32 %16, 1
%i328B

	full_text
	
i32 %16
2add8B)
'
	full_text

%29 = add i32 %28, %14
%i328B

	full_text
	
i32 %28
%i328B

	full_text
	
i32 %14
6sext8B,
*
	full_text

%30 = sext i32 %29 to i64
%i328B

	full_text
	
i32 %29
\getelementptr8BI
G
	full_text:
8
6%31 = getelementptr inbounds float, float* %0, i64 %30
%i648B

	full_text
	
i64 %30
@bitcast8B3
1
	full_text$
"
 %32 = bitcast float* %31 to i32*
+float*8B

	full_text


float* %31
Hload8B>
<
	full_text/
-
+%33 = load i32, i32* %32, align 4, !tbaa !8
'i32*8B

	full_text


i32* %32
1shl8B(
&
	full_text

%34 = shl i64 %13, 32
%i648B

	full_text
	
i64 %13
9ashr8B/
-
	full_text 

%35 = ashr exact i64 %34, 32
%i648B

	full_text
	
i64 %34
\getelementptr8BI
G
	full_text:
8
6%36 = getelementptr inbounds float, float* %4, i64 %35
%i648B

	full_text
	
i64 %35
@bitcast8B3
1
	full_text$
"
 %37 = bitcast float* %36 to i32*
+float*8B

	full_text


float* %36
Hstore8B=
;
	full_text.
,
*store i32 %33, i32* %37, align 4, !tbaa !8
%i328B

	full_text
	
i32 %33
'i32*8B

	full_text


i32* %37
'br8B

	full_text

br label %38
Gphi8B>
<
	full_text/
-
+%39 = phi float* [ %26, %23 ], [ %36, %27 ]
+float*8B

	full_text


float* %26
+float*8B

	full_text


float* %36
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
6sext8B,
*
	full_text

%40 = sext i32 %21 to i64
%i328B

	full_text
	
i32 %21
\getelementptr8BI
G
	full_text:
8
6%41 = getelementptr inbounds float, float* %2, i64 %40
%i648B

	full_text
	
i64 %40
@bitcast8B3
1
	full_text$
"
 %42 = bitcast float* %41 to i32*
+float*8B

	full_text


float* %41
Hload8B>
<
	full_text/
-
+%43 = load i32, i32* %42, align 4, !tbaa !8
'i32*8B

	full_text


i32* %42
0shl8B'
%
	full_text

%44 = shl i32 %14, 4
%i328B

	full_text
	
i32 %14
6add8B-
+
	full_text

%45 = add nsw i32 %44, %12
%i328B

	full_text
	
i32 %44
%i328B

	full_text
	
i32 %12
6sext8B,
*
	full_text

%46 = sext i32 %45 to i64
%i328B

	full_text
	
i32 %45
\getelementptr8BI
G
	full_text:
8
6%47 = getelementptr inbounds float, float* %5, i64 %46
%i648B

	full_text
	
i64 %46
@bitcast8B3
1
	full_text$
"
 %48 = bitcast float* %47 to i32*
+float*8B

	full_text


float* %47
Hstore8B=
;
	full_text.
,
*store i32 %43, i32* %48, align 4, !tbaa !8
%i328B

	full_text
	
i32 %43
'i32*8B

	full_text


i32* %48
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
Lload8BB
@
	full_text3
1
/%49 = load float, float* %47, align 4, !tbaa !8
+float*8B

	full_text


float* %47
Lload8BB
@
	full_text3
1
/%50 = load float, float* %39, align 4, !tbaa !8
+float*8B

	full_text


float* %39
6fmul8B,
*
	full_text

%51 = fmul float %49, %50
)float8B

	full_text

	float %49
)float8B

	full_text

	float %50
Lstore8BA
?
	full_text2
0
.store float %51, float* %47, align 4, !tbaa !8
)float8B

	full_text

	float %51
+float*8B

	full_text


float* %47
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
Lload8BB
@
	full_text3
1
/%52 = load float, float* %47, align 4, !tbaa !8
+float*8B

	full_text


float* %47
6fadd8B,
*
	full_text

%53 = fadd float %52, %52
)float8B

	full_text

	float %52
)float8B

	full_text

	float %52
Lstore8BA
?
	full_text2
0
.store float %53, float* %47, align 4, !tbaa !8
)float8B

	full_text

	float %53
+float*8B

	full_text


float* %47
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
0and8B'
%
	full_text

%54 = and i32 %14, 1
%i328B

	full_text
	
i32 %14
5icmp8B+
)
	full_text

%55 = icmp eq i32 %54, 0
%i328B

	full_text
	
i32 %54
:br8B2
0
	full_text#
!
br i1 %55, label %68, label %77
#i18B

	full_text


i1 %55
1shl8B(
&
	full_text

%57 = shl i64 %13, 32
%i648B

	full_text
	
i64 %13
9ashr8B/
-
	full_text 

%58 = ashr exact i64 %57, 32
%i648B

	full_text
	
i64 %57
\getelementptr8BI
G
	full_text:
8
6%59 = getelementptr inbounds float, float* %5, i64 %58
%i648B

	full_text
	
i64 %58
@bitcast8B3
1
	full_text$
"
 %60 = bitcast float* %59 to i32*
+float*8B

	full_text


float* %59
Hload8B>
<
	full_text/
-
+%61 = load i32, i32* %60, align 4, !tbaa !8
'i32*8B

	full_text


i32* %60
5mul8B,
*
	full_text

%62 = mul nsw i32 %10, %7
%i328B

	full_text
	
i32 %10
6add8B-
+
	full_text

%63 = add nsw i32 %62, %14
%i328B

	full_text
	
i32 %62
%i328B

	full_text
	
i32 %14
6sext8B,
*
	full_text

%64 = sext i32 %63 to i64
%i328B

	full_text
	
i32 %63
\getelementptr8BI
G
	full_text:
8
6%65 = getelementptr inbounds float, float* %3, i64 %64
%i648B

	full_text
	
i64 %64
@bitcast8B3
1
	full_text$
"
 %66 = bitcast float* %65 to i32*
+float*8B

	full_text


float* %65
Hstore8B=
;
	full_text.
,
*store i32 %61, i32* %66, align 4, !tbaa !8
%i328B

	full_text
	
i32 %61
'i32*8B

	full_text


i32* %66
'br8B

	full_text

br label %67
$ret8B

	full_text


ret void
Lload8BB
@
	full_text3
1
/%69 = load float, float* %47, align 4, !tbaa !8
+float*8B

	full_text


float* %47
0shl8B'
%
	full_text

%70 = shl i32 %14, 4
%i328B

	full_text
	
i32 %14
1add8B(
&
	full_text

%71 = add i32 %70, 16
%i328B

	full_text
	
i32 %70
6add8B-
+
	full_text

%72 = add nsw i32 %71, %12
%i328B

	full_text
	
i32 %71
%i328B

	full_text
	
i32 %12
6sext8B,
*
	full_text

%73 = sext i32 %72 to i64
%i328B

	full_text
	
i32 %72
\getelementptr8BI
G
	full_text:
8
6%74 = getelementptr inbounds float, float* %5, i64 %73
%i648B

	full_text
	
i64 %73
Lload8BB
@
	full_text3
1
/%75 = load float, float* %74, align 4, !tbaa !8
+float*8B

	full_text


float* %74
6fadd8B,
*
	full_text

%76 = fadd float %69, %75
)float8B

	full_text

	float %69
)float8B

	full_text

	float %75
Lstore8BA
?
	full_text2
0
.store float %76, float* %47, align 4, !tbaa !8
)float8B

	full_text

	float %76
+float*8B

	full_text


float* %47
'br8B

	full_text

br label %77
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
0and8B'
%
	full_text

%78 = and i32 %14, 3
%i328B

	full_text
	
i32 %14
5icmp8B+
)
	full_text

%79 = icmp eq i32 %78, 0
%i328B

	full_text
	
i32 %78
:br8B2
0
	full_text#
!
br i1 %79, label %80, label %89
#i18B

	full_text


i1 %79
Lload8BB
@
	full_text3
1
/%81 = load float, float* %47, align 4, !tbaa !8
+float*8B

	full_text


float* %47
0shl8B'
%
	full_text

%82 = shl i32 %14, 4
%i328B

	full_text
	
i32 %14
1add8B(
&
	full_text

%83 = add i32 %82, 32
%i328B

	full_text
	
i32 %82
6add8B-
+
	full_text

%84 = add nsw i32 %83, %12
%i328B

	full_text
	
i32 %83
%i328B

	full_text
	
i32 %12
6sext8B,
*
	full_text

%85 = sext i32 %84 to i64
%i328B

	full_text
	
i32 %84
\getelementptr8BI
G
	full_text:
8
6%86 = getelementptr inbounds float, float* %5, i64 %85
%i648B

	full_text
	
i64 %85
Lload8BB
@
	full_text3
1
/%87 = load float, float* %86, align 4, !tbaa !8
+float*8B

	full_text


float* %86
6fadd8B,
*
	full_text

%88 = fadd float %81, %87
)float8B

	full_text

	float %81
)float8B

	full_text

	float %87
Lstore8BA
?
	full_text2
0
.store float %88, float* %47, align 4, !tbaa !8
)float8B

	full_text

	float %88
+float*8B

	full_text


float* %47
'br8B

	full_text

br label %89
Bcall8	B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
0and8	B'
%
	full_text

%90 = and i32 %14, 7
%i328	B

	full_text
	
i32 %14
5icmp8	B+
)
	full_text

%91 = icmp eq i32 %90, 0
%i328	B

	full_text
	
i32 %90
;br8	B3
1
	full_text$
"
 br i1 %91, label %92, label %101
#i18	B

	full_text


i1 %91
Lload8
BB
@
	full_text3
1
/%93 = load float, float* %47, align 4, !tbaa !8
+float*8
B

	full_text


float* %47
0shl8
B'
%
	full_text

%94 = shl i32 %14, 4
%i328
B

	full_text
	
i32 %14
1add8
B(
&
	full_text

%95 = add i32 %94, 64
%i328
B

	full_text
	
i32 %94
6add8
B-
+
	full_text

%96 = add nsw i32 %95, %12
%i328
B

	full_text
	
i32 %95
%i328
B

	full_text
	
i32 %12
6sext8
B,
*
	full_text

%97 = sext i32 %96 to i64
%i328
B

	full_text
	
i32 %96
\getelementptr8
BI
G
	full_text:
8
6%98 = getelementptr inbounds float, float* %5, i64 %97
%i648
B

	full_text
	
i64 %97
Lload8
BB
@
	full_text3
1
/%99 = load float, float* %98, align 4, !tbaa !8
+float*8
B

	full_text


float* %98
7fadd8
B-
+
	full_text

%100 = fadd float %93, %99
)float8
B

	full_text

	float %93
)float8
B

	full_text

	float %99
Mstore8
BB
@
	full_text3
1
/store float %100, float* %47, align 4, !tbaa !8
*float8
B

	full_text


float %100
+float*8
B

	full_text


float* %47
(br8
B 

	full_text

br label %101
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
2and8B)
'
	full_text

%102 = and i32 %14, 15
%i328B

	full_text
	
i32 %14
7icmp8B-
+
	full_text

%103 = icmp eq i32 %102, 0
&i328B

	full_text


i32 %102
=br8B5
3
	full_text&
$
"br i1 %103, label %104, label %113
$i18B

	full_text
	
i1 %103
Mload8BC
A
	full_text4
2
0%105 = load float, float* %47, align 4, !tbaa !8
+float*8B

	full_text


float* %47
1shl8B(
&
	full_text

%106 = shl i32 %14, 4
%i328B

	full_text
	
i32 %14
4add8B+
)
	full_text

%107 = add i32 %106, 128
&i328B

	full_text


i32 %106
8add8B/
-
	full_text 

%108 = add nsw i32 %107, %12
&i328B

	full_text


i32 %107
%i328B

	full_text
	
i32 %12
8sext8B.
,
	full_text

%109 = sext i32 %108 to i64
&i328B

	full_text


i32 %108
^getelementptr8BK
I
	full_text<
:
8%110 = getelementptr inbounds float, float* %5, i64 %109
&i648B

	full_text


i64 %109
Nload8BD
B
	full_text5
3
1%111 = load float, float* %110, align 4, !tbaa !8
,float*8B

	full_text

float* %110
9fadd8B/
-
	full_text 

%112 = fadd float %105, %111
*float8B

	full_text


float %105
*float8B

	full_text


float %111
Mstore8BB
@
	full_text3
1
/store float %112, float* %47, align 4, !tbaa !8
*float8B

	full_text


float %112
+float*8B

	full_text


float* %47
(br8B 

	full_text

br label %113
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
Iload8B?
=
	full_text0
.
,%114 = load i32, i32* %48, align 4, !tbaa !8
'i32*8B

	full_text


i32* %48
Istore8B>
<
	full_text/
-
+store i32 %114, i32* %42, align 4, !tbaa !8
&i328B

	full_text


i32 %114
'i32*8B

	full_text


i32* %42
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
:br8B2
0
	full_text#
!
br i1 %22, label %56, label %67
#i18B

	full_text


i1 %22
*float*8B

	full_text

	float* %5
*float*8B

	full_text

	float* %3
*float*8B

	full_text

	float* %4
$i328B

	full_text


i32 %7
*float*8B

	full_text

	float* %2
*float*8B
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
#i328B

	full_text	

i32 3
#i328B

	full_text	

i32 1
$i328B

	full_text


i32 32
$i328B

	full_text


i32 16
#i328B

	full_text	

i32 7
%i328B

	full_text
	
i32 128
$i328B

	full_text


i32 15
#i328B

	full_text	

i32 4
#i328B

	full_text	

i32 0
$i648B

	full_text


i64 32
#i328B

	full_text	

i32 2
$i328B

	full_text


i32 64       	  

                      !    "# "" $& %% '( ') '' *+ ** ,- ,, ./ .. 01 00 23 22 45 44 67 66 89 88 :; :< :: =? >@ >> AA BC BB DE DD FG FF HI HH JK JJ LM LN LL OP OO QR QQ ST SS UV UW UU XX YZ YY [\ [[ ]^ ]_ ]] `a `b `` cc de dd fg fh ff ij ik ii ll mn mm op oo qr qt ss uv uu wx ww yz yy {| {{ }~ }} Ä 	Å  ÇÉ ÇÇ Ñ
Ö ÑÑ Üá ÜÜ àâ à
ä àà ãé çç èê èè ëí ëë ìî ì
ï ìì ñó ññ ò
ô òò öõ öö úù ú
û úú ü† ü
° üü ¢£ §• §§ ¶ß ¶¶ ®© ®´ ™™ ¨≠ ¨¨ ÆØ ÆÆ ∞± ∞
≤ ∞∞ ≥¥ ≥≥ µ
∂ µµ ∑∏ ∑∑ π∫ π
ª ππ ºΩ º
æ ºº ø¿ ¡¬ ¡¡ √ƒ √√ ≈∆ ≈» «« …  …… ÀÃ ÀÀ ÕŒ Õ
œ ÕÕ –— –– “
” ““ ‘’ ‘‘ ÷◊ ÷
ÿ ÷÷ Ÿ⁄ Ÿ
€ ŸŸ ‹› ﬁﬂ ﬁﬁ ‡· ‡‡ ‚„ ‚Â ‰‰ ÊÁ ÊÊ ËÈ ËË ÍÎ Í
Ï ÍÍ ÌÓ ÌÌ Ô
 ÔÔ ÒÚ ÒÒ ÛÙ Û
ı ÛÛ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˚¸ ˚˚ ˝˛ ˝
ˇ ˝˝ ÄÄ ÅÇ ÅÉ QÉ wÉ òÉ µÉ “É ÔÑ ÑÖ "Ö 6Ü 
Ü 	Ü }á Dà ,   	    
         !  # &% ( )' +* -, /. 1 32 54 76 90 ;8 <" ?6 @ CB ED GF I KJ M NL PO RQ TH VS WQ Z> \Y ^[ _] aQ bQ ed gd hf jQ k nm po r ts vu xw zy | ~} Ä Å ÉÇ ÖÑ á{ âÜ äQ é êè íë î ïì óñ ôò õç ùö ûú †Q ° •§ ß¶ ©Q ´ ≠¨ ØÆ ± ≤∞ ¥≥ ∂µ ∏™ ∫∑ ªπ ΩQ æ ¬¡ ƒ√ ∆Q »  … ÃÀ Œ œÕ —– ”“ ’« ◊‘ ÿ÷ ⁄Q € ﬂﬁ ·‡ „Q Â ÁÊ ÈË Î ÏÍ ÓÌ Ô Ú‰ ÙÒ ıÛ ˜Q ¯S ¸˚ ˛F ˇ Ç % = >$ >q çq £¢ £® ™® ¿ø ¿≈ «≈ ›‹ ›‚ ‰‚ ˙˘ ˙Å sÅ åã å ââ ää å ãã ââ X ãã X¿ ãã ¿Ä ãã ÄA ãã Ac ãã c› ãã ›l ãã l˙ ãã ˙ ää  ää £ ãã £
å §ç ç 	ç 
	ç %ç Aç Xç cç l	ç mç £ç ¿ç ›ç ˙ç Ä
é Æ
è ë
ê ¡
ë Ë
í ﬁ	ì 	ì J
ì è
ì ¨
ì …
ì Êî 	î 	î o
î ¶
î √
î ‡	ï 	ï  	ï 2	ï 4	ï s	ï u	ñ 
ó À"
bpnn_layerforward_ocl"
_Z12get_group_idj"
_Z12get_local_idj"
_Z7barrierj*¶
-rodinia-3.1-backprop-bpnn_layerforward_ocl.clu
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

transfer_bytes
îÇ¿

wgsize_log1p
I{ÄA
 
transfer_bytes_log1p
I{ÄA

wgsize
Ä