

[external]
KcallBC
A
	full_text4
2
0%7 = tail call i64 @_Z13get_global_idj(i32 0) #2
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
McallBE
C
	full_text6
4
2%9 = tail call i64 @_Z15get_global_sizej(i32 0) #2
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
.shlB'
%
	full_text

%11 = shl i64 %7, 32
"i64B

	full_text


i64 %7
7ashrB/
-
	full_text 

%12 = ashr exact i64 %11, 32
#i64B

	full_text
	
i64 %11
VgetelementptrBE
C
	full_text6
4
2%13 = getelementptr inbounds i32, i32* %2, i64 %12
#i64B

	full_text
	
i64 %12
FloadB>
<
	full_text/
-
+%14 = load i32, i32* %13, align 4, !tbaa !8
%i32*B

	full_text


i32* %13
5icmpB-
+
	full_text

%15 = icmp slt i32 %14, %4
#i32B

	full_text
	
i32 %14
8brB2
0
	full_text#
!
br i1 %15, label %16, label %97
!i1B

	full_text


i1 %15
Xgetelementptr8BE
C
	full_text6
4
2%17 = getelementptr inbounds i32, i32* %3, i64 %12
%i648B

	full_text
	
i64 %12
Hload8B>
<
	full_text/
-
+%18 = load i32, i32* %17, align 4, !tbaa !8
'i32*8B

	full_text


i32* %17
5sext8B+
)
	full_text

%19 = sext i32 %5 to i64
Xgetelementptr8BE
C
	full_text6
4
2%20 = getelementptr inbounds i32, i32* %0, i64 %19
%i648B

	full_text
	
i64 %19
4add8B+
)
	full_text

%21 = add nsw i32 %14, 1
%i328B

	full_text
	
i32 %14
8icmp8B.
,
	full_text

%22 = icmp slt i32 %21, %18
%i328B

	full_text
	
i32 %21
%i328B

	full_text
	
i32 %18
:br8B2
0
	full_text#
!
br i1 %22, label %23, label %52
#i18B

	full_text


i1 %22
6sext8B,
*
	full_text

%24 = sext i32 %14 to i64
%i328B

	full_text
	
i32 %14
4add8B+
)
	full_text

%25 = add nsw i64 %24, 1
%i648B

	full_text
	
i64 %24
0add8B'
%
	full_text

%26 = add i32 %18, 3
%i328B

	full_text
	
i32 %18
2sub8B)
'
	full_text

%27 = sub i32 %26, %14
%i328B

	full_text
	
i32 %26
%i328B

	full_text
	
i32 %14
1add8B(
&
	full_text

%28 = add i32 %18, -2
%i328B

	full_text
	
i32 %18
2sub8B)
'
	full_text

%29 = sub i32 %28, %14
%i328B

	full_text
	
i32 %28
%i328B

	full_text
	
i32 %14
0and8B'
%
	full_text

%30 = and i32 %27, 3
%i328B

	full_text
	
i32 %27
5icmp8B+
)
	full_text

%31 = icmp eq i32 %30, 0
%i328B

	full_text
	
i32 %30
:br8B2
0
	full_text#
!
br i1 %31, label %47, label %32
#i18B

	full_text


i1 %31
'br8B

	full_text

br label %33
Dphi8B;
9
	full_text,
*
(%34 = phi i64 [ %25, %32 ], [ %43, %33 ]
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %43
Dphi8B;
9
	full_text,
*
(%35 = phi i32 [ %14, %32 ], [ %44, %33 ]
%i328B

	full_text
	
i32 %14
%i328B

	full_text
	
i32 %44
Dphi8B;
9
	full_text,
*
(%36 = phi i32 [ %30, %32 ], [ %45, %33 ]
%i328B

	full_text
	
i32 %30
%i328B

	full_text
	
i32 %45
Ygetelementptr8BF
D
	full_text7
5
3%37 = getelementptr inbounds i32, i32* %20, i64 %34
'i32*8B

	full_text


i32* %20
%i648B

	full_text
	
i64 %34
Hload8B>
<
	full_text/
-
+%38 = load i32, i32* %37, align 4, !tbaa !8
'i32*8B

	full_text


i32* %37
6sext8B,
*
	full_text

%39 = sext i32 %35 to i64
%i328B

	full_text
	
i32 %35
Ygetelementptr8BF
D
	full_text7
5
3%40 = getelementptr inbounds i32, i32* %20, i64 %39
'i32*8B

	full_text


i32* %20
%i648B

	full_text
	
i64 %39
Hload8B>
<
	full_text/
-
+%41 = load i32, i32* %40, align 4, !tbaa !8
'i32*8B

	full_text


i32* %40
6add8B-
+
	full_text

%42 = add nsw i32 %41, %38
%i328B

	full_text
	
i32 %41
%i328B

	full_text
	
i32 %38
Hstore8B=
;
	full_text.
,
*store i32 %42, i32* %37, align 4, !tbaa !8
%i328B

	full_text
	
i32 %42
'i32*8B

	full_text


i32* %37
0add8B'
%
	full_text

%43 = add i64 %34, 1
%i648B

	full_text
	
i64 %34
8trunc8B-
+
	full_text

%44 = trunc i64 %34 to i32
%i648B

	full_text
	
i64 %34
1add8B(
&
	full_text

%45 = add i32 %36, -1
%i328B

	full_text
	
i32 %36
5icmp8B+
)
	full_text

%46 = icmp eq i32 %45, 0
%i328B

	full_text
	
i32 %45
Jbr8BB
@
	full_text3
1
/br i1 %46, label %47, label %33, !llvm.loop !12
#i18B

	full_text


i1 %46
Dphi8B;
9
	full_text,
*
(%48 = phi i64 [ %25, %23 ], [ %43, %33 ]
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %43
Dphi8B;
9
	full_text,
*
(%49 = phi i32 [ %14, %23 ], [ %44, %33 ]
%i328B

	full_text
	
i32 %14
%i328B

	full_text
	
i32 %44
6icmp8B,
*
	full_text

%50 = icmp ult i32 %29, 3
%i328B

	full_text
	
i32 %29
:br8B2
0
	full_text#
!
br i1 %50, label %52, label %51
#i18B

	full_text


i1 %50
'br8B

	full_text

br label %54
7icmp8B-
+
	full_text

%53 = icmp slt i32 %8, %10
$i328B

	full_text


i32 %8
%i328B

	full_text
	
i32 %10
:br8B2
0
	full_text#
!
br i1 %53, label %91, label %97
#i18B

	full_text


i1 %53
Dphi8B;
9
	full_text,
*
(%55 = phi i64 [ %48, %51 ], [ %87, %54 ]
%i648B

	full_text
	
i64 %48
%i648B

	full_text
	
i64 %87
Dphi8B;
9
	full_text,
*
(%56 = phi i32 [ %49, %51 ], [ %88, %54 ]
%i328B

	full_text
	
i32 %49
%i328B

	full_text
	
i32 %88
Ygetelementptr8BF
D
	full_text7
5
3%57 = getelementptr inbounds i32, i32* %20, i64 %55
'i32*8B

	full_text


i32* %20
%i648B

	full_text
	
i64 %55
Hload8B>
<
	full_text/
-
+%58 = load i32, i32* %57, align 4, !tbaa !8
'i32*8B

	full_text


i32* %57
6sext8B,
*
	full_text

%59 = sext i32 %56 to i64
%i328B

	full_text
	
i32 %56
Ygetelementptr8BF
D
	full_text7
5
3%60 = getelementptr inbounds i32, i32* %20, i64 %59
'i32*8B

	full_text


i32* %20
%i648B

	full_text
	
i64 %59
Hload8B>
<
	full_text/
-
+%61 = load i32, i32* %60, align 4, !tbaa !8
'i32*8B

	full_text


i32* %60
6add8B-
+
	full_text

%62 = add nsw i32 %61, %58
%i328B

	full_text
	
i32 %61
%i328B

	full_text
	
i32 %58
Hstore8B=
;
	full_text.
,
*store i32 %62, i32* %57, align 4, !tbaa !8
%i328B

	full_text
	
i32 %62
'i32*8B

	full_text


i32* %57
0add8B'
%
	full_text

%63 = add i64 %55, 1
%i648B

	full_text
	
i64 %55
Ygetelementptr8BF
D
	full_text7
5
3%64 = getelementptr inbounds i32, i32* %20, i64 %63
'i32*8B

	full_text


i32* %20
%i648B

	full_text
	
i64 %63
Hload8B>
<
	full_text/
-
+%65 = load i32, i32* %64, align 4, !tbaa !8
'i32*8B

	full_text


i32* %64
1shl8B(
&
	full_text

%66 = shl i64 %55, 32
%i648B

	full_text
	
i64 %55
9ashr8B/
-
	full_text 

%67 = ashr exact i64 %66, 32
%i648B

	full_text
	
i64 %66
Ygetelementptr8BF
D
	full_text7
5
3%68 = getelementptr inbounds i32, i32* %20, i64 %67
'i32*8B

	full_text


i32* %20
%i648B

	full_text
	
i64 %67
Hload8B>
<
	full_text/
-
+%69 = load i32, i32* %68, align 4, !tbaa !8
'i32*8B

	full_text


i32* %68
6add8B-
+
	full_text

%70 = add nsw i32 %69, %65
%i328B

	full_text
	
i32 %69
%i328B

	full_text
	
i32 %65
Hstore8B=
;
	full_text.
,
*store i32 %70, i32* %64, align 4, !tbaa !8
%i328B

	full_text
	
i32 %70
'i32*8B

	full_text


i32* %64
0add8B'
%
	full_text

%71 = add i64 %55, 2
%i648B

	full_text
	
i64 %55
Ygetelementptr8BF
D
	full_text7
5
3%72 = getelementptr inbounds i32, i32* %20, i64 %71
'i32*8B

	full_text


i32* %20
%i648B

	full_text
	
i64 %71
Hload8B>
<
	full_text/
-
+%73 = load i32, i32* %72, align 4, !tbaa !8
'i32*8B

	full_text


i32* %72
1shl8B(
&
	full_text

%74 = shl i64 %63, 32
%i648B

	full_text
	
i64 %63
9ashr8B/
-
	full_text 

%75 = ashr exact i64 %74, 32
%i648B

	full_text
	
i64 %74
Ygetelementptr8BF
D
	full_text7
5
3%76 = getelementptr inbounds i32, i32* %20, i64 %75
'i32*8B

	full_text


i32* %20
%i648B

	full_text
	
i64 %75
Hload8B>
<
	full_text/
-
+%77 = load i32, i32* %76, align 4, !tbaa !8
'i32*8B

	full_text


i32* %76
6add8B-
+
	full_text

%78 = add nsw i32 %77, %73
%i328B

	full_text
	
i32 %77
%i328B

	full_text
	
i32 %73
Hstore8B=
;
	full_text.
,
*store i32 %78, i32* %72, align 4, !tbaa !8
%i328B

	full_text
	
i32 %78
'i32*8B

	full_text


i32* %72
0add8B'
%
	full_text

%79 = add i64 %55, 3
%i648B

	full_text
	
i64 %55
Ygetelementptr8BF
D
	full_text7
5
3%80 = getelementptr inbounds i32, i32* %20, i64 %79
'i32*8B

	full_text


i32* %20
%i648B

	full_text
	
i64 %79
Hload8B>
<
	full_text/
-
+%81 = load i32, i32* %80, align 4, !tbaa !8
'i32*8B

	full_text


i32* %80
1shl8B(
&
	full_text

%82 = shl i64 %71, 32
%i648B

	full_text
	
i64 %71
9ashr8B/
-
	full_text 

%83 = ashr exact i64 %82, 32
%i648B

	full_text
	
i64 %82
Ygetelementptr8BF
D
	full_text7
5
3%84 = getelementptr inbounds i32, i32* %20, i64 %83
'i32*8B

	full_text


i32* %20
%i648B

	full_text
	
i64 %83
Hload8B>
<
	full_text/
-
+%85 = load i32, i32* %84, align 4, !tbaa !8
'i32*8B

	full_text


i32* %84
6add8B-
+
	full_text

%86 = add nsw i32 %85, %81
%i328B

	full_text
	
i32 %85
%i328B

	full_text
	
i32 %81
Hstore8B=
;
	full_text.
,
*store i32 %86, i32* %80, align 4, !tbaa !8
%i328B

	full_text
	
i32 %86
'i32*8B

	full_text


i32* %80
0add8B'
%
	full_text

%87 = add i64 %55, 4
%i648B

	full_text
	
i64 %55
8trunc8B-
+
	full_text

%88 = trunc i64 %79 to i32
%i648B

	full_text
	
i64 %79
8trunc8B-
+
	full_text

%89 = trunc i64 %87 to i32
%i648B

	full_text
	
i64 %87
7icmp8B-
+
	full_text

%90 = icmp eq i32 %18, %89
%i328B

	full_text
	
i32 %18
%i328B

	full_text
	
i32 %89
:br8B2
0
	full_text#
!
br i1 %90, label %52, label %54
#i18B

	full_text


i1 %90
5add8	B,
*
	full_text

%92 = add nsw i32 %18, -1
%i328	B

	full_text
	
i32 %18
6sext8	B,
*
	full_text

%93 = sext i32 %92 to i64
%i328	B

	full_text
	
i32 %92
Ygetelementptr8	BF
D
	full_text7
5
3%94 = getelementptr inbounds i32, i32* %20, i64 %93
'i32*8	B

	full_text


i32* %20
%i648	B

	full_text
	
i64 %93
Hload8	B>
<
	full_text/
-
+%95 = load i32, i32* %94, align 4, !tbaa !8
'i32*8	B

	full_text


i32* %94
Xgetelementptr8	BE
C
	full_text6
4
2%96 = getelementptr inbounds i32, i32* %1, i64 %12
%i648	B

	full_text
	
i64 %12
Hstore8	B=
;
	full_text.
,
*store i32 %95, i32* %96, align 4, !tbaa !8
%i328	B

	full_text
	
i32 %95
'i32*8	B

	full_text


i32* %96
'br8	B

	full_text

br label %97
$ret8
B

	full_text


ret void
&i32*8B

	full_text
	
i32* %0
&i32*8B

	full_text
	
i32* %3
$i328B

	full_text


i32 %5
&i32*8B

	full_text
	
i32* %1
&i32*8B

	full_text
	
i32* %2
$i328B
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
$i328B

	full_text


i32 -2
$i328B

	full_text


i32 -1
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 4
#i328B

	full_text	

i32 3
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 2
#i328B

	full_text	

i32 1
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 3        	
 		                      " !! #$ ## %& %% '( ') '' *+ ** ,- ,. ,, /0 // 12 11 34 37 68 66 9: 9; 99 <= <> << ?@ ?A ?? BC BB DE DD FG FH FF IJ II KL KM KK NO NP NN QR QQ ST SS UV UU WX WW YZ Y\ [] [[ ^_ ^` ^^ ab aa cd cg fh ff ij il km kk no np nn qr qs qq tu tt vw vv xy xz xx {| {{ }~ } }} € €
‚ €€ ƒ„ ƒƒ …† …
‡ …… ˆ‰ ˆˆ Š‹ ŠŠ Œ ŒŒ Ž Ž
 ŽŽ ‘’ ‘‘ “” “
• ““ –— –
˜ –– ™š ™™ ›œ ›
 ›› žŸ žž  ¡    ¢£ ¢¢ ¤¥ ¤
¦ ¤¤ §¨ §§ ©ª ©
« ©© ¬­ ¬
® ¬¬ ¯° ¯¯ ±² ±
³ ±± ´µ ´´ ¶· ¶¶ ¸¹ ¸¸ º» º
¼ ºº ½¾ ½½ ¿À ¿
Á ¿¿ ÂÃ Â
Ä ÂÂ ÅÆ ÅÅ ÇÈ ÇÇ ÉÊ ÉÉ ËÌ Ë
Í ËË ÎÏ ÎÑ ÐÐ ÒÓ ÒÒ ÔÕ Ô
Ö ÔÔ ×Ø ×× Ù
Ú ÙÙ ÛÜ Û
Ý ÛÛ Þà á â ã Ùä 	å     
	    	         "! $ &% ( ) +* - .' 0/ 21 4# 7Q 8 :S ;/ =U > @6 A? C9 E GD HF JI LB MK O? P6 R6 T< VU XW Z# \Q ] _S `, ba d g hf j[ lÅ m^ oÇ p rk sq un w yv zx |{ ~t } q ‚k „ †ƒ ‡… ‰k ‹Š  Œ Ž ’‘ ”ˆ •“ —… ˜k š œ™ › Ÿƒ ¡  £ ¥¢ ¦¤ ¨§ ªž «© ­› ®k ° ²¯ ³± µ™ ·¶ ¹ »¸ ¼º ¾½ À´ Á¿ Ã± Äk Æ¯ ÈÅ Ê ÌÉ ÍË Ï ÑÐ Ó ÕÒ ÖÔ Ø	 Ú× ÜÙ Ý  ß ! f3 [3 5i Ði ßc fc e5 6Þ ße kY [Y 6Î fÎ k çç ß ææ ææ  çç 	è *	é U
é Ðê ê 	ê 1	ê W
ë Å	ì %	ì /	ì a	í #	í Q
í ƒ
î ™	ï 	ð 	ð 	
ð Š
ð Œ
ð  
ð ¢
ð ¶
ð ¸
ñ ¯"	
makea_4"
_Z13get_global_idj"
_Z15get_global_sizej*Š
npb-CG-makea_4.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282
 
transfer_bytes_log1p
PÑA

wgsize
 

devmap_label
 

wgsize_log1p
PÑA

transfer_bytes	
¼—Ž°