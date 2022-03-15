

[external]
KcallBC
A
	full_text4
2
0%8 = tail call i64 @_Z13get_global_idj(i32 0) #2
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
4icmpB,
*
	full_text

%10 = icmp slt i32 %9, %6
"i32B

	full_text


i32 %9
9brB3
1
	full_text$
"
 br i1 %10, label %11, label %112
!i1B

	full_text


i1 %10
5sext8B+
)
	full_text

%12 = sext i32 %5 to i64
Xgetelementptr8BE
C
	full_text6
4
2%13 = getelementptr inbounds i32, i32* %4, i64 %12
%i648B

	full_text
	
i64 %12
5icmp8B+
)
	full_text

%14 = icmp sgt i32 %9, 0
$i328B

	full_text


i32 %9
0shl8B'
%
	full_text

%15 = shl i64 %8, 32
$i648B

	full_text


i64 %8
9ashr8B/
-
	full_text 

%16 = ashr exact i64 %15, 32
%i648B

	full_text
	
i64 %15
:br8B2
0
	full_text#
!
br i1 %14, label %17, label %25
#i18B

	full_text


i1 %14
Xgetelementptr8BE
C
	full_text6
4
2%18 = getelementptr inbounds i32, i32* %2, i64 %16
%i648B

	full_text
	
i64 %16
Hload8B>
<
	full_text/
-
+%19 = load i32, i32* %18, align 4, !tbaa !8
'i32*8B

	full_text


i32* %18
:add8B1
/
	full_text"
 
%20 = add i64 %15, -4294967296
%i648B

	full_text
	
i64 %15
9ashr8B/
-
	full_text 

%21 = ashr exact i64 %20, 32
%i648B

	full_text
	
i64 %20
Ygetelementptr8BF
D
	full_text7
5
3%22 = getelementptr inbounds i32, i32* %13, i64 %21
'i32*8B

	full_text


i32* %13
%i648B

	full_text
	
i64 %21
Hload8B>
<
	full_text/
-
+%23 = load i32, i32* %22, align 4, !tbaa !8
'i32*8B

	full_text


i32* %22
6sub8B-
+
	full_text

%24 = sub nsw i32 %19, %23
%i328B

	full_text
	
i32 %19
%i328B

	full_text
	
i32 %23
'br8B

	full_text

br label %25
Bphi8B9
7
	full_text*
(
&%26 = phi i32 [ %24, %17 ], [ 0, %11 ]
%i328B

	full_text
	
i32 %24
9add8B0
.
	full_text!

%27 = add i64 %15, 4294967296
%i648B

	full_text
	
i64 %15
9ashr8B/
-
	full_text 

%28 = ashr exact i64 %27, 32
%i648B

	full_text
	
i64 %27
Xgetelementptr8BE
C
	full_text6
4
2%29 = getelementptr inbounds i32, i32* %2, i64 %28
%i648B

	full_text
	
i64 %28
Hload8B>
<
	full_text/
-
+%30 = load i32, i32* %29, align 4, !tbaa !8
'i32*8B

	full_text


i32* %29
Ygetelementptr8BF
D
	full_text7
5
3%31 = getelementptr inbounds i32, i32* %13, i64 %16
'i32*8B

	full_text


i32* %13
%i648B

	full_text
	
i64 %16
Hload8B>
<
	full_text/
-
+%32 = load i32, i32* %31, align 4, !tbaa !8
'i32*8B

	full_text


i32* %31
6sub8B-
+
	full_text

%33 = sub nsw i32 %30, %32
%i328B

	full_text
	
i32 %30
%i328B

	full_text
	
i32 %32
8icmp8B.
,
	full_text

%34 = icmp slt i32 %26, %33
%i328B

	full_text
	
i32 %26
%i328B

	full_text
	
i32 %33
;br8B3
1
	full_text$
"
 br i1 %34, label %35, label %112
#i18B

	full_text


i1 %34
Xgetelementptr8BE
C
	full_text6
4
2%36 = getelementptr inbounds i32, i32* %2, i64 %16
%i648B

	full_text
	
i64 %16
Hload8B>
<
	full_text/
-
+%37 = load i32, i32* %36, align 4, !tbaa !8
'i32*8B

	full_text


i32* %36
6sext8B,
*
	full_text

%38 = sext i32 %26 to i64
%i328B

	full_text
	
i32 %26
6sext8B,
*
	full_text

%39 = sext i32 %37 to i64
%i328B

	full_text
	
i32 %37
6sext8B,
*
	full_text

%40 = sext i32 %33 to i64
%i328B

	full_text
	
i32 %33
6sub8B-
+
	full_text

%41 = sub nsw i64 %40, %38
%i648B

	full_text
	
i64 %40
%i648B

	full_text
	
i64 %38
5add8B,
*
	full_text

%42 = add nsw i64 %40, -1
%i648B

	full_text
	
i64 %40
6sub8B-
+
	full_text

%43 = sub nsw i64 %42, %38
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %38
0and8B'
%
	full_text

%44 = and i64 %41, 3
%i648B

	full_text
	
i64 %41
5icmp8B+
)
	full_text

%45 = icmp eq i64 %44, 0
%i648B

	full_text
	
i64 %44
:br8B2
0
	full_text#
!
br i1 %45, label %63, label %46
#i18B

	full_text


i1 %45
'br8B

	full_text

br label %47
Dphi8B;
9
	full_text,
*
(%48 = phi i64 [ %39, %46 ], [ %59, %47 ]
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %59
Dphi8B;
9
	full_text,
*
(%49 = phi i64 [ %38, %46 ], [ %60, %47 ]
%i648B

	full_text
	
i64 %38
%i648B

	full_text
	
i64 %60
Dphi8B;
9
	full_text,
*
(%50 = phi i64 [ %44, %46 ], [ %61, %47 ]
%i648B

	full_text
	
i64 %44
%i648B

	full_text
	
i64 %61
^getelementptr8BK
I
	full_text<
:
8%51 = getelementptr inbounds double, double* %1, i64 %48
%i648B

	full_text
	
i64 %48
Abitcast8B4
2
	full_text%
#
!%52 = bitcast double* %51 to i64*
-double*8B

	full_text

double* %51
Iload8B?
=
	full_text0
.
,%53 = load i64, i64* %52, align 8, !tbaa !12
'i64*8B

	full_text


i64* %52
^getelementptr8BK
I
	full_text<
:
8%54 = getelementptr inbounds double, double* %0, i64 %49
%i648B

	full_text
	
i64 %49
Abitcast8B4
2
	full_text%
#
!%55 = bitcast double* %54 to i64*
-double*8B

	full_text

double* %54
Istore8B>
<
	full_text/
-
+store i64 %53, i64* %55, align 8, !tbaa !12
%i648B

	full_text
	
i64 %53
'i64*8B

	full_text


i64* %55
Xgetelementptr8BE
C
	full_text6
4
2%56 = getelementptr inbounds i32, i32* %4, i64 %48
%i648B

	full_text
	
i64 %48
Hload8B>
<
	full_text/
-
+%57 = load i32, i32* %56, align 4, !tbaa !8
'i32*8B

	full_text


i32* %56
Xgetelementptr8BE
C
	full_text6
4
2%58 = getelementptr inbounds i32, i32* %3, i64 %49
%i648B

	full_text
	
i64 %49
Hstore8B=
;
	full_text.
,
*store i32 %57, i32* %58, align 4, !tbaa !8
%i328B

	full_text
	
i32 %57
'i32*8B

	full_text


i32* %58
4add8B+
)
	full_text

%59 = add nsw i64 %48, 1
%i648B

	full_text
	
i64 %48
4add8B+
)
	full_text

%60 = add nsw i64 %49, 1
%i648B

	full_text
	
i64 %49
1add8B(
&
	full_text

%61 = add i64 %50, -1
%i648B

	full_text
	
i64 %50
5icmp8B+
)
	full_text

%62 = icmp eq i64 %61, 0
%i648B

	full_text
	
i64 %61
Jbr8BB
@
	full_text3
1
/br i1 %62, label %63, label %47, !llvm.loop !14
#i18B

	full_text


i1 %62
Dphi8B;
9
	full_text,
*
(%64 = phi i64 [ %39, %35 ], [ %59, %47 ]
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %59
Dphi8B;
9
	full_text,
*
(%65 = phi i64 [ %38, %35 ], [ %60, %47 ]
%i648B

	full_text
	
i64 %38
%i648B

	full_text
	
i64 %60
6icmp8B,
*
	full_text

%66 = icmp ult i64 %43, 3
%i648B

	full_text
	
i64 %43
;br8B3
1
	full_text$
"
 br i1 %66, label %112, label %67
#i18B

	full_text


i1 %66
'br8B

	full_text

br label %68
Ephi8	B<
:
	full_text-
+
)%69 = phi i64 [ %64, %67 ], [ %109, %68 ]
%i648	B

	full_text
	
i64 %64
&i648	B

	full_text


i64 %109
Ephi8	B<
:
	full_text-
+
)%70 = phi i64 [ %65, %67 ], [ %110, %68 ]
%i648	B

	full_text
	
i64 %65
&i648	B

	full_text


i64 %110
^getelementptr8	BK
I
	full_text<
:
8%71 = getelementptr inbounds double, double* %1, i64 %69
%i648	B

	full_text
	
i64 %69
Abitcast8	B4
2
	full_text%
#
!%72 = bitcast double* %71 to i64*
-double*8	B

	full_text

double* %71
Iload8	B?
=
	full_text0
.
,%73 = load i64, i64* %72, align 8, !tbaa !12
'i64*8	B

	full_text


i64* %72
^getelementptr8	BK
I
	full_text<
:
8%74 = getelementptr inbounds double, double* %0, i64 %70
%i648	B

	full_text
	
i64 %70
Abitcast8	B4
2
	full_text%
#
!%75 = bitcast double* %74 to i64*
-double*8	B

	full_text

double* %74
Istore8	B>
<
	full_text/
-
+store i64 %73, i64* %75, align 8, !tbaa !12
%i648	B

	full_text
	
i64 %73
'i64*8	B

	full_text


i64* %75
Xgetelementptr8	BE
C
	full_text6
4
2%76 = getelementptr inbounds i32, i32* %4, i64 %69
%i648	B

	full_text
	
i64 %69
Hload8	B>
<
	full_text/
-
+%77 = load i32, i32* %76, align 4, !tbaa !8
'i32*8	B

	full_text


i32* %76
Xgetelementptr8	BE
C
	full_text6
4
2%78 = getelementptr inbounds i32, i32* %3, i64 %70
%i648	B

	full_text
	
i64 %70
Hstore8	B=
;
	full_text.
,
*store i32 %77, i32* %78, align 4, !tbaa !8
%i328	B

	full_text
	
i32 %77
'i32*8	B

	full_text


i32* %78
4add8	B+
)
	full_text

%79 = add nsw i64 %69, 1
%i648	B

	full_text
	
i64 %69
4add8	B+
)
	full_text

%80 = add nsw i64 %70, 1
%i648	B

	full_text
	
i64 %70
^getelementptr8	BK
I
	full_text<
:
8%81 = getelementptr inbounds double, double* %1, i64 %79
%i648	B

	full_text
	
i64 %79
Abitcast8	B4
2
	full_text%
#
!%82 = bitcast double* %81 to i64*
-double*8	B

	full_text

double* %81
Iload8	B?
=
	full_text0
.
,%83 = load i64, i64* %82, align 8, !tbaa !12
'i64*8	B

	full_text


i64* %82
^getelementptr8	BK
I
	full_text<
:
8%84 = getelementptr inbounds double, double* %0, i64 %80
%i648	B

	full_text
	
i64 %80
Abitcast8	B4
2
	full_text%
#
!%85 = bitcast double* %84 to i64*
-double*8	B

	full_text

double* %84
Istore8	B>
<
	full_text/
-
+store i64 %83, i64* %85, align 8, !tbaa !12
%i648	B

	full_text
	
i64 %83
'i64*8	B

	full_text


i64* %85
Xgetelementptr8	BE
C
	full_text6
4
2%86 = getelementptr inbounds i32, i32* %4, i64 %79
%i648	B

	full_text
	
i64 %79
Hload8	B>
<
	full_text/
-
+%87 = load i32, i32* %86, align 4, !tbaa !8
'i32*8	B

	full_text


i32* %86
Xgetelementptr8	BE
C
	full_text6
4
2%88 = getelementptr inbounds i32, i32* %3, i64 %80
%i648	B

	full_text
	
i64 %80
Hstore8	B=
;
	full_text.
,
*store i32 %87, i32* %88, align 4, !tbaa !8
%i328	B

	full_text
	
i32 %87
'i32*8	B

	full_text


i32* %88
4add8	B+
)
	full_text

%89 = add nsw i64 %69, 2
%i648	B

	full_text
	
i64 %69
4add8	B+
)
	full_text

%90 = add nsw i64 %70, 2
%i648	B

	full_text
	
i64 %70
^getelementptr8	BK
I
	full_text<
:
8%91 = getelementptr inbounds double, double* %1, i64 %89
%i648	B

	full_text
	
i64 %89
Abitcast8	B4
2
	full_text%
#
!%92 = bitcast double* %91 to i64*
-double*8	B

	full_text

double* %91
Iload8	B?
=
	full_text0
.
,%93 = load i64, i64* %92, align 8, !tbaa !12
'i64*8	B

	full_text


i64* %92
^getelementptr8	BK
I
	full_text<
:
8%94 = getelementptr inbounds double, double* %0, i64 %90
%i648	B

	full_text
	
i64 %90
Abitcast8	B4
2
	full_text%
#
!%95 = bitcast double* %94 to i64*
-double*8	B

	full_text

double* %94
Istore8	B>
<
	full_text/
-
+store i64 %93, i64* %95, align 8, !tbaa !12
%i648	B

	full_text
	
i64 %93
'i64*8	B

	full_text


i64* %95
Xgetelementptr8	BE
C
	full_text6
4
2%96 = getelementptr inbounds i32, i32* %4, i64 %89
%i648	B

	full_text
	
i64 %89
Hload8	B>
<
	full_text/
-
+%97 = load i32, i32* %96, align 4, !tbaa !8
'i32*8	B

	full_text


i32* %96
Xgetelementptr8	BE
C
	full_text6
4
2%98 = getelementptr inbounds i32, i32* %3, i64 %90
%i648	B

	full_text
	
i64 %90
Hstore8	B=
;
	full_text.
,
*store i32 %97, i32* %98, align 4, !tbaa !8
%i328	B

	full_text
	
i32 %97
'i32*8	B

	full_text


i32* %98
4add8	B+
)
	full_text

%99 = add nsw i64 %69, 3
%i648	B

	full_text
	
i64 %69
5add8	B,
*
	full_text

%100 = add nsw i64 %70, 3
%i648	B

	full_text
	
i64 %70
_getelementptr8	BL
J
	full_text=
;
9%101 = getelementptr inbounds double, double* %1, i64 %99
%i648	B

	full_text
	
i64 %99
Cbitcast8	B6
4
	full_text'
%
#%102 = bitcast double* %101 to i64*
.double*8	B

	full_text

double* %101
Kload8	BA
?
	full_text2
0
.%103 = load i64, i64* %102, align 8, !tbaa !12
(i64*8	B

	full_text

	i64* %102
`getelementptr8	BM
K
	full_text>
<
:%104 = getelementptr inbounds double, double* %0, i64 %100
&i648	B

	full_text


i64 %100
Cbitcast8	B6
4
	full_text'
%
#%105 = bitcast double* %104 to i64*
.double*8	B

	full_text

double* %104
Kstore8	B@
>
	full_text1
/
-store i64 %103, i64* %105, align 8, !tbaa !12
&i648	B

	full_text


i64 %103
(i64*8	B

	full_text

	i64* %105
Ygetelementptr8	BF
D
	full_text7
5
3%106 = getelementptr inbounds i32, i32* %4, i64 %99
%i648	B

	full_text
	
i64 %99
Jload8	B@
>
	full_text1
/
-%107 = load i32, i32* %106, align 4, !tbaa !8
(i32*8	B

	full_text

	i32* %106
Zgetelementptr8	BG
E
	full_text8
6
4%108 = getelementptr inbounds i32, i32* %3, i64 %100
&i648	B

	full_text


i64 %100
Jstore8	B?
=
	full_text0
.
,store i32 %107, i32* %108, align 4, !tbaa !8
&i328	B

	full_text


i32 %107
(i32*8	B

	full_text

	i32* %108
5add8	B,
*
	full_text

%109 = add nsw i64 %69, 4
%i648	B

	full_text
	
i64 %69
5add8	B,
*
	full_text

%110 = add nsw i64 %70, 4
%i648	B

	full_text
	
i64 %70
9icmp8	B/
-
	full_text 

%111 = icmp eq i64 %110, %40
&i648	B

	full_text


i64 %110
%i648	B

	full_text
	
i64 %40
<br8	B4
2
	full_text%
#
!br i1 %111, label %112, label %68
$i18	B

	full_text
	
i1 %111
$ret8
B

	full_text


ret void
,double*8B

	full_text


double* %0
,double*8B

	full_text


double* %1
$i328B

	full_text


i32 %6
&i32*8B

	full_text
	
i32* %3
&i32*8B

	full_text
	
i32* %4
$i328B

	full_text


i32 %5
&i32*8B

	full_text
	
i32* %2
-; undefined function B

	full_text

 
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 3
$i648B

	full_text


i64 -1
,i648B!

	full_text

i64 4294967296
$i648B

	full_text


i64 32
-i648B"
 
	full_text

i64 -4294967296
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 4
#i648B

	full_text	

i64 2       	
 		                      !  "    #% $$ &' && () (( *+ ** ,- ,, ./ .0 .. 12 11 34 35 33 67 68 66 9: 9< ;; => == ?@ ?? AB AA CD CC EF EG EE HI HH JK JL JJ MN MM OP OO QR QU TV TT WX WY WW Z[ Z\ ZZ ]^ ]] _` __ ab aa cd cc ef ee gh gi gg jk jj lm ll no nn pq pr pp st ss uv uu wx ww yz yy {| {~ } }} ÄÅ Ä
Ç ÄÄ ÉÑ ÉÉ ÖÜ Öâ à
ä àà ãå ã
ç ãã é
è éé êë êê íì íí î
ï îî ñó ññ òô ò
ö òò õ
ú õõ ùû ùù ü
† üü °¢ °
£ °° §• §§ ¶ß ¶¶ ®
© ®® ™´ ™™ ¨≠ ¨¨ Æ
Ø ÆÆ ∞± ∞∞ ≤≥ ≤
¥ ≤≤ µ
∂ µµ ∑∏ ∑∑ π
∫ ππ ªº ª
Ω ªª æø ææ ¿¡ ¿¿ ¬
√ ¬¬ ƒ≈ ƒƒ ∆« ∆∆ »
… »»  À    ÃÕ Ã
Œ ÃÃ œ
– œœ —“ —— ”
‘ ”” ’÷ ’
◊ ’’ ÿŸ ÿÿ ⁄€ ⁄⁄ ‹
› ‹‹ ﬁﬂ ﬁﬁ ‡· ‡‡ ‚
„ ‚‚ ‰Â ‰‰ ÊÁ Ê
Ë ÊÊ È
Í ÈÈ ÎÏ ÎÎ Ì
Ó ÌÌ Ô Ô
Ò ÔÔ ÚÛ ÚÚ Ùı ÙÙ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘¸ c¸ î¸ Æ¸ »¸ ‚˝ ]˝ é˝ ®˝ ¬˝ ‹	˛ ˇ nˇ üˇ πˇ ”ˇ ÌÄ 	Ä jÄ õÄ µÄ œÄ ÈÅ Ç Ç *Ç ;    
        	    ! "  % '& )( +* -	 / 0. 2, 41 5$ 73 86 : <; >$ @= B3 DC F? GC IH K? LE NM PO RA Us V? Xu YM [w \T ^] `_ bW dc fa he iT kj mW ol qn rT tW vZ xw zy |A ~s ? Åu ÇJ ÑÉ Ü} âÚ äÄ åÙ çà èé ëê ìã ïî óí ôñ öà úõ ûã †ù ¢ü £à •ã ß§ ©® ´™ ≠¶ ØÆ ±¨ ≥∞ ¥§ ∂µ ∏¶ ∫∑ ºπ Ωà øã ¡æ √¬ ≈ƒ «¿ …» À∆ Õ  Œæ –œ “¿ ‘— ÷” ◊à Ÿã €ÿ ›‹ ﬂﬁ ·⁄ „‚ Â‡ Á‰ Ëÿ ÍÈ Ï⁄ ÓÎ Ì Òà Ûã ıÙ ˜C ¯ˆ ˙  ˚  $# $9 ;9 ˚Q }Q SÖ ˚Ö áS Tá à{ }{ T˘ ˚˘ à ˚ ÉÉ ÉÉ Ñ 	Ñ 	Ñ $	Ö O	Ö y	Ü M
Ü É
Ü ÿ
Ü ⁄	á H	á w	à &	â 	â 	â 	â (	ä 	ã s	ã u
ã §
ã ¶
å Ú
å Ù
ç æ
ç ¿"	
makea_6"
_Z13get_global_idj*ä
npb-CG-makea_6.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Å

wgsize_log1p
Q éA

wgsize
Ä

transfer_bytes
‹…±
 
transfer_bytes_log1p
Q éA

devmap_label
 