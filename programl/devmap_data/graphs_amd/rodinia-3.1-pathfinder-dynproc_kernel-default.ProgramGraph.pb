

[external]
McallBE
C
	full_text6
4
2%13 = tail call i64 @_Z14get_local_sizej(i32 0) #3
6truncB-
+
	full_text

%14 = trunc i64 %13 to i32
#i64B

	full_text
	
i64 %13
KcallBC
A
	full_text4
2
0%15 = tail call i64 @_Z12get_group_idj(i32 0) #3
6truncB-
+
	full_text

%16 = trunc i64 %15 to i32
#i64B

	full_text
	
i64 %15
KcallBC
A
	full_text4
2
0%17 = tail call i64 @_Z12get_local_idj(i32 0) #3
6truncB-
+
	full_text

%18 = trunc i64 %17 to i32
#i64B

	full_text
	
i64 %17
-shlB&
$
	full_text

%19 = shl i32 %0, 1
/mulB(
&
	full_text

%20 = mul i32 %19, %8
#i32B

	full_text
	
i32 %19
4subB-
+
	full_text

%21 = sub nsw i32 %14, %20
#i32B

	full_text
	
i32 %14
#i32B

	full_text
	
i32 %20
4mulB-
+
	full_text

%22 = mul nsw i32 %21, %16
#i32B

	full_text
	
i32 %21
#i32B

	full_text
	
i32 %16
3subB,
*
	full_text

%23 = sub nsw i32 %22, %7
#i32B

	full_text
	
i32 %22
4addB-
+
	full_text

%24 = add nsw i32 %23, %14
#i32B

	full_text
	
i32 %23
#i32B

	full_text
	
i32 %14
4addB-
+
	full_text

%25 = add nsw i32 %23, %18
#i32B

	full_text
	
i32 %23
#i32B

	full_text
	
i32 %18
4icmpB,
*
	full_text

%26 = icmp slt i32 %23, 0
#i32B

	full_text
	
i32 %23
2subB+
)
	full_text

%27 = sub nsw i32 0, %23
#i32B

	full_text
	
i32 %23
@selectB6
4
	full_text'
%
#%28 = select i1 %26, i32 %27, i32 0
!i1B

	full_text


i1 %26
#i32B

	full_text
	
i32 %27
5icmpB-
+
	full_text

%29 = icmp sgt i32 %24, %4
#i32B

	full_text
	
i32 %24
3addB,
*
	full_text

%30 = add nsw i32 %14, -1
#i32B

	full_text
	
i32 %14
/addB(
&
	full_text

%31 = add i32 %30, %4
#i32B

	full_text
	
i32 %30
0subB)
'
	full_text

%32 = sub i32 %31, %24
#i32B

	full_text
	
i32 %31
#i32B

	full_text
	
i32 %24
BselectB8
6
	full_text)
'
%%33 = select i1 %29, i32 %32, i32 %30
!i1B

	full_text


i1 %29
#i32B

	full_text
	
i32 %32
#i32B

	full_text
	
i32 %30
3addB,
*
	full_text

%34 = add nsw i32 %18, -1
#i32B

	full_text
	
i32 %18
2addB+
)
	full_text

%35 = add nsw i32 %18, 1
#i32B

	full_text
	
i32 %18
6icmpB.
,
	full_text

%36 = icmp slt i32 %34, %28
#i32B

	full_text
	
i32 %34
#i32B

	full_text
	
i32 %28
BselectB8
6
	full_text)
'
%%37 = select i1 %36, i32 %28, i32 %34
!i1B

	full_text


i1 %36
#i32B

	full_text
	
i32 %28
#i32B

	full_text
	
i32 %34
6icmpB.
,
	full_text

%38 = icmp sgt i32 %35, %33
#i32B

	full_text
	
i32 %35
#i32B

	full_text
	
i32 %33
BselectB8
6
	full_text)
'
%%39 = select i1 %38, i32 %33, i32 %35
!i1B

	full_text


i1 %38
#i32B

	full_text
	
i32 %33
#i32B

	full_text
	
i32 %35
6icmpB.
,
	full_text

%40 = icmp sgt i32 %28, %18
#i32B

	full_text
	
i32 %28
#i32B

	full_text
	
i32 %18
6icmpB.
,
	full_text

%41 = icmp slt i32 %33, %18
#i32B

	full_text
	
i32 %33
#i32B

	full_text
	
i32 %18
5icmpB-
+
	full_text

%42 = icmp sgt i32 %25, -1
#i32B

	full_text
	
i32 %25
5icmpB-
+
	full_text

%43 = icmp slt i32 %25, %4
#i32B

	full_text
	
i32 %25
/andB(
&
	full_text

%44 = and i1 %42, %43
!i1B

	full_text


i1 %42
!i1B

	full_text


i1 %43
8brB2
0
	full_text#
!
br i1 %44, label %45, label %52
!i1B

	full_text


i1 %44
6sext8B,
*
	full_text

%46 = sext i32 %25 to i64
%i328B

	full_text
	
i32 %25
Xgetelementptr8BE
C
	full_text6
4
2%47 = getelementptr inbounds i32, i32* %2, i64 %46
%i648B

	full_text
	
i64 %46
Hload8B>
<
	full_text/
-
+%48 = load i32, i32* %47, align 4, !tbaa !8
'i32*8B

	full_text


i32* %47
1shl8B(
&
	full_text

%49 = shl i64 %17, 32
%i648B

	full_text
	
i64 %17
9ashr8B/
-
	full_text 

%50 = ashr exact i64 %49, 32
%i648B

	full_text
	
i64 %49
Xgetelementptr8BE
C
	full_text6
4
2%51 = getelementptr inbounds i32, i32* %9, i64 %50
%i648B

	full_text
	
i64 %50
Hstore8B=
;
	full_text.
,
*store i32 %48, i32* %51, align 4, !tbaa !8
%i328B

	full_text
	
i32 %48
'i32*8B

	full_text


i32* %51
'br8B

	full_text

br label %52
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
5icmp8B+
)
	full_text

%53 = icmp sgt i32 %0, 0
;br8B3
1
	full_text$
"
 br i1 %53, label %54, label %122
#i18B

	full_text


i1 %53
1add8B(
&
	full_text

%55 = add i32 %14, -2
%i328B

	full_text
	
i32 %14
/or8B'
%
	full_text

%56 = or i1 %40, %41
#i18B

	full_text


i1 %40
#i18B

	full_text


i1 %41
6sext8B,
*
	full_text

%57 = sext i32 %37 to i64
%i328B

	full_text
	
i32 %37
Xgetelementptr8BE
C
	full_text6
4
2%58 = getelementptr inbounds i32, i32* %9, i64 %57
%i648B

	full_text
	
i64 %57
1shl8B(
&
	full_text

%59 = shl i64 %17, 32
%i648B

	full_text
	
i64 %17
9ashr8B/
-
	full_text 

%60 = ashr exact i64 %59, 32
%i648B

	full_text
	
i64 %59
Xgetelementptr8BE
C
	full_text6
4
2%61 = getelementptr inbounds i32, i32* %9, i64 %60
%i648B

	full_text
	
i64 %60
6sext8B,
*
	full_text

%62 = sext i32 %39 to i64
%i328B

	full_text
	
i32 %39
Xgetelementptr8BE
C
	full_text6
4
2%63 = getelementptr inbounds i32, i32* %9, i64 %62
%i648B

	full_text
	
i64 %62
Ygetelementptr8BF
D
	full_text7
5
3%64 = getelementptr inbounds i32, i32* %10, i64 %60
%i648B

	full_text
	
i64 %60
6icmp8B,
*
	full_text

%65 = icmp eq i32 %18, 11
%i328B

	full_text
	
i32 %18
6sext8B,
*
	full_text

%66 = sext i32 %25 to i64
%i328B

	full_text
	
i32 %25
Xgetelementptr8BE
C
	full_text6
4
2%67 = getelementptr inbounds i32, i32* %2, i64 %66
%i648B

	full_text
	
i64 %66
4add8B+
)
	full_text

%68 = add nsw i32 %0, -1
5sext8B+
)
	full_text

%69 = sext i32 %6 to i64
5sext8B+
)
	full_text

%70 = sext i32 %4 to i64
1shl8B(
&
	full_text

%71 = shl i64 %17, 32
%i648B

	full_text
	
i64 %17
9ashr8B/
-
	full_text 

%72 = ashr exact i64 %71, 32
%i648B

	full_text
	
i64 %71
6zext8B,
*
	full_text

%73 = zext i32 %68 to i64
%i328B

	full_text
	
i32 %68
5sext8B+
)
	full_text

%74 = sext i32 %0 to i64
'br8B

	full_text

br label %75
Cphi8B:
8
	full_text+
)
'%76 = phi i64 [ 0, %54 ], [ %77, %111 ]
%i648B

	full_text
	
i64 %77
8add8B/
-
	full_text 

%77 = add nuw nsw i64 %76, 1
%i648B

	full_text
	
i64 %76
8icmp8B.
,
	full_text

%78 = icmp slt i64 %76, %72
%i648B

	full_text
	
i64 %76
%i648B

	full_text
	
i64 %72
;br8B3
1
	full_text$
"
 br i1 %78, label %79, label %104
#i18B

	full_text


i1 %78
8trunc8B-
+
	full_text

%80 = trunc i64 %76 to i32
%i648B

	full_text
	
i64 %76
2sub8B)
'
	full_text

%81 = sub i32 %55, %80
%i328B

	full_text
	
i32 %55
%i328B

	full_text
	
i32 %80
8icmp8B.
,
	full_text

%82 = icmp slt i32 %81, %18
%i328B

	full_text
	
i32 %81
%i328B

	full_text
	
i32 %18
/or8B'
%
	full_text

%83 = or i1 %56, %82
#i18B

	full_text


i1 %56
#i18B

	full_text


i1 %82
;br8B3
1
	full_text$
"
 br i1 %83, label %104, label %84
#i18B

	full_text


i1 %83
Hload8B>
<
	full_text/
-
+%85 = load i32, i32* %58, align 4, !tbaa !8
'i32*8B

	full_text


i32* %58
Hload8B>
<
	full_text/
-
+%86 = load i32, i32* %61, align 4, !tbaa !8
'i32*8B

	full_text


i32* %61
Hload8B>
<
	full_text/
-
+%87 = load i32, i32* %63, align 4, !tbaa !8
'i32*8B

	full_text


i32* %63
8icmp8B.
,
	full_text

%88 = icmp sgt i32 %85, %86
%i328B

	full_text
	
i32 %85
%i328B

	full_text
	
i32 %86
Dselect8B8
6
	full_text)
'
%%89 = select i1 %88, i32 %86, i32 %85
#i18B

	full_text


i1 %88
%i328B

	full_text
	
i32 %86
%i328B

	full_text
	
i32 %85
8icmp8B.
,
	full_text

%90 = icmp sgt i32 %89, %87
%i328B

	full_text
	
i32 %89
%i328B

	full_text
	
i32 %87
Dselect8B8
6
	full_text)
'
%%91 = select i1 %90, i32 %87, i32 %89
#i18B

	full_text


i1 %90
%i328B

	full_text
	
i32 %87
%i328B

	full_text
	
i32 %89
6add8B-
+
	full_text

%92 = add nsw i64 %76, %69
%i648B

	full_text
	
i64 %76
%i648B

	full_text
	
i64 %69
6mul8B-
+
	full_text

%93 = mul nsw i64 %92, %70
%i648B

	full_text
	
i64 %92
%i648B

	full_text
	
i64 %70
6add8B-
+
	full_text

%94 = add nsw i64 %93, %66
%i648B

	full_text
	
i64 %93
%i648B

	full_text
	
i64 %66
Xgetelementptr8BE
C
	full_text6
4
2%95 = getelementptr inbounds i32, i32* %1, i64 %94
%i648B

	full_text
	
i64 %94
Hload8B>
<
	full_text/
-
+%96 = load i32, i32* %95, align 4, !tbaa !8
'i32*8B

	full_text


i32* %95
6add8B-
+
	full_text

%97 = add nsw i32 %91, %96
%i328B

	full_text
	
i32 %91
%i328B

	full_text
	
i32 %96
Hstore8B=
;
	full_text.
,
*store i32 %97, i32* %64, align 4, !tbaa !8
%i328B

	full_text
	
i32 %97
'i32*8B

	full_text


i32* %64
5icmp8B+
)
	full_text

%98 = icmp eq i64 %76, 0
%i648B

	full_text
	
i64 %76
1and8B(
&
	full_text

%99 = and i1 %65, %98
#i18B

	full_text


i1 %65
#i18B

	full_text


i1 %98
<br8B4
2
	full_text%
#
!br i1 %99, label %100, label %104
#i18B

	full_text


i1 %99
Iload8B?
=
	full_text0
.
,%101 = load i32, i32* %67, align 4, !tbaa !8
'i32*8B

	full_text


i32* %67
8sext8B.
,
	full_text

%102 = sext i32 %101 to i64
&i328B

	full_text


i32 %101
[getelementptr8BH
F
	full_text9
7
5%103 = getelementptr inbounds i32, i32* %11, i64 %102
&i648B

	full_text


i64 %102
Gstore8B<
:
	full_text-
+
)store i32 1, i32* %103, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %103
(br8B 

	full_text

br label %104
Yphi8BP
N
	full_textA
?
=%105 = phi i8 [ 0, %79 ], [ 0, %75 ], [ 1, %100 ], [ 1, %84 ]
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
8icmp8B.
,
	full_text

%106 = icmp eq i64 %76, %73
%i648B

	full_text
	
i64 %76
%i648B

	full_text
	
i64 %73
=br8B5
3
	full_text&
$
"br i1 %106, label %113, label %107
$i18B

	full_text
	
i1 %106
6icmp8	B,
*
	full_text

%108 = icmp eq i8 %105, 0
$i88	B

	full_text
	
i8 %105
=br8	B5
3
	full_text&
$
"br i1 %108, label %111, label %109
$i18	B

	full_text
	
i1 %108
Iload8
B?
=
	full_text0
.
,%110 = load i32, i32* %64, align 4, !tbaa !8
'i32*8
B

	full_text


i32* %64
Istore8
B>
<
	full_text/
-
+store i32 %110, i32* %61, align 4, !tbaa !8
&i328
B

	full_text


i32 %110
'i32*8
B

	full_text


i32* %61
(br8
B 

	full_text

br label %111
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
9icmp8B/
-
	full_text 

%112 = icmp slt i64 %77, %74
%i648B

	full_text
	
i64 %77
%i648B

	full_text
	
i64 %74
<br8B4
2
	full_text%
#
!br i1 %112, label %75, label %113
$i18B

	full_text
	
i1 %112
6icmp8B,
*
	full_text

%114 = icmp eq i8 %105, 0
$i88B

	full_text
	
i8 %105
=br8B5
3
	full_text&
$
"br i1 %114, label %122, label %115
$i18B

	full_text
	
i1 %114
2shl8B)
'
	full_text

%116 = shl i64 %17, 32
%i648B

	full_text
	
i64 %17
;ashr8B1
/
	full_text"
 
%117 = ashr exact i64 %116, 32
&i648B

	full_text


i64 %116
[getelementptr8BH
F
	full_text9
7
5%118 = getelementptr inbounds i32, i32* %10, i64 %117
&i648B

	full_text


i64 %117
Jload8B@
>
	full_text1
/
-%119 = load i32, i32* %118, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %118
7sext8B-
+
	full_text

%120 = sext i32 %25 to i64
%i328B

	full_text
	
i32 %25
Zgetelementptr8BG
E
	full_text8
6
4%121 = getelementptr inbounds i32, i32* %3, i64 %120
&i648B

	full_text


i64 %120
Jstore8B?
=
	full_text0
.
,store i32 %119, i32* %121, align 4, !tbaa !8
&i328B

	full_text


i32 %119
(i32*8B

	full_text

	i32* %121
(br8B 

	full_text

br label %122
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %0
&i32*8B

	full_text
	
i32* %2
$i328B

	full_text


i32 %6
$i328B

	full_text


i32 %8
'i32*8B

	full_text


i32* %11
&i32*8B

	full_text
	
i32* %9
$i328B

	full_text


i32 %4
&i32*8B

	full_text
	
i32* %1
'i32*8B

	full_text


i32* %10
&i32*8B

	full_text
	
i32* %3
$i328B

	full_text


i32 %7
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
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 0
!i88B

	full_text

i8 0
$i328B

	full_text


i32 -2
$i648B

	full_text


i64 32
!i88B

	full_text

i8 1
#i648B

	full_text	

i64 1
$i328B

	full_text


i32 11
$i328B

	full_text


i32 -1
#i328B

	full_text	

i32 0       	  

                       !  "# "" $% $$ &' && () (* (( +, +- +. ++ /0 // 12 11 34 35 33 67 68 69 66 :; :< :: => =? =@ == AB AC AA DE DF DD GH GG IJ II KL KM KK NO NQ PP RS RR TU TT VW VV XY XX Z[ ZZ \] \^ \\ _` aa bc be dd fg fh ff ij ii kl kk mn mm op oo qr qq st ss uv uu wx ww yz yy {| {{ }~ }}  ÄÄ ÅÅ ÇÉ ÇÇ ÑÖ ÑÑ Üá ÜÜ àà â
ã ää åç åå éè é
ê éé ëí ëî ìì ïñ ï
ó ïï òô ò
ö òò õú õ
ù õõ ûü û° †† ¢£ ¢¢ §• §§ ¶ß ¶
® ¶¶ ©™ ©
´ ©
¨ ©© ≠Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞
≥ ∞∞ ¥µ ¥
∂ ¥¥ ∑∏ ∑
π ∑∑ ∫ª ∫
º ∫∫ Ω
æ ΩΩ ø¿ øø ¡¬ ¡
√ ¡¡ ƒ≈ ƒ
∆ ƒƒ «» «« …  …
À …… ÃÕ Ãœ ŒŒ –— –– “
” ““ ‘
’ ‘‘ ÷◊ ÿÿ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹ﬂ ﬁﬁ ‡· ‡„ ‚‚ ‰Â ‰
Ê ‰‰ ÁË ÈÍ È
Î ÈÈ ÏÌ ÏÔ ÓÓ Ò Û ÚÚ Ùı ÙÙ ˆ
˜ ˆˆ ¯˘ ¯¯ ˙˚ ˙˙ ¸
˝ ¸¸ ˛ˇ ˛
Ä ˛˛ ÅÉ 
É aÉ É àÑ RÑ }Ö Ä	Ü á “à Zà kà qà u	â "	â &	â Iâ Åä Ωã wã ˆå ¸	ç    	
               ! # %$ '& ) *" ,( -$ . 0 2/ 4 53 7 8/ 91 ;+ <: >+ ?1 @ B C+ E F H JG LI MK O QP SR U WV YX [T ]Z ^a c eA gD h6 ji l nm po r= ts vo x z |{ ~ ÉÇ Ö áå ãä çä èÑ êé íä îd ñì óï ô öf úò ùõ ük °q £u •† ß¢ ®¶ ™¢ ´† ¨© Æ§ Ø≠ ±§ ≤© ≥ä µÄ ∂¥ ∏Å π∑ ª{ º∫ æΩ ¿∞ ¬ø √¡ ≈w ∆ä »y  « À… Õ} œŒ —– ”“ ’ä ⁄Ü €Ÿ ›◊ ﬂﬁ ·w „‚ Âq Êå Íà ÎÈ Ì◊ ÔÓ Ò ÛÚ ıÙ ˜ˆ ˘ ˚˙ ˝¯ ˇ¸ ÄN PN `_ `b db Çâ äë ìë ◊û ◊û †‹ Ó‹ ﬁÃ ŒÃ ◊ Ç Ú‡ Ë‡ ‚÷ ◊Å ÇÏ äÏ ÓÁ Ë Ç ëë êê éé èè èè ÿ ëë ÿË ëë Ë` ëë ` éé  êê 	í 
	í 1í `í ‘í ÿí Ëì ä
ì «î ◊
î ◊
î ﬁ
î Ó	ï d	ñ V	ñ X	ñ m	ñ o
ñ Ç
ñ Ñ
ñ Ú
ñ Ù
ó ◊
ó ◊
ò å	ô y	ö $	ö /	ö G	ö õ õ õ 	õ õ 	õ 	õ a"
dynproc_kernel"
_Z14get_local_sizej"
_Z12get_group_idj"
_Z12get_local_idj"
_Z7barrierj*°
(rodinia-3.1-pathfinder-dynproc_kernel.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Å

wgsize_log1p
∞ åA

devmap_label


transfer_bytes
ÄÈ•

wgsize
Ò
 
transfer_bytes_log1p
∞ åA