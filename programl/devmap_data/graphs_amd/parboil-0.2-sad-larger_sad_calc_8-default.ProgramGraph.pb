

[external]
JcallBB
@
	full_text3
1
/%4 = tail call i64 @_Z12get_local_idj(i32 1) #2
4truncB+
)
	full_text

%5 = trunc i64 %4 to i32
"i64B

	full_text


i64 %4
,andB%
#
	full_text

%6 = and i32 %5, 1
"i32B

	full_text


i32 %5
.lshrB&
$
	full_text

%7 = lshr i64 %4, 1
"i64B

	full_text


i64 %4
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
JcallBB
@
	full_text3
1
/%9 = tail call i64 @_Z12get_group_idj(i32 0) #2
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
0%11 = tail call i64 @_Z12get_group_idj(i32 1) #2
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
0%13 = tail call i64 @_Z12get_local_idj(i32 0) #2
6truncB-
+
	full_text

%14 = trunc i64 %13 to i32
#i64B

	full_text
	
i64 %13
McallBE
C
	full_text6
4
2%15 = tail call i32 @_Z5mul24ii(i32 %1, i32 %2) #2
NcallBF
D
	full_text7
5
3%16 = tail call i32 @_Z5mul24ii(i32 %12, i32 %1) #2
#i32B

	full_text
	
i32 %12
4addB-
+
	full_text

%17 = add nsw i32 %16, %10
#i32B

	full_text
	
i32 %16
#i32B

	full_text
	
i32 %10
3zextB+
)
	full_text

%18 = zext i32 %6 to i64
"i32B

	full_text


i32 %6
=inttoptrB1
/
	full_text"
 
%19 = inttoptr i64 %18 to i16*
#i64B

	full_text
	
i64 %18
NcallBF
D
	full_text7
5
3%20 = tail call i32 @_Z5mul24ii(i32 %15, i32 25) #2
#i32B

	full_text
	
i32 %15
-shlB&
$
	full_text

%21 = shl i64 %7, 3
"i64B

	full_text


i64 %7
6truncB-
+
	full_text

%22 = trunc i64 %21 to i32
#i64B

	full_text
	
i64 %21
5shlB.
,
	full_text

%23 = shl nuw nsw i32 %6, 1
"i32B

	full_text


i32 %6
.orB(
&
	full_text

%24 = or i32 %23, %22
#i32B

	full_text
	
i32 %23
#i32B

	full_text
	
i32 %22
4addB-
+
	full_text

%25 = add nsw i32 %20, %24
#i32B

	full_text
	
i32 %20
#i32B

	full_text
	
i32 %24
5mulB.
,
	full_text

%26 = mul nsw i32 %25, 1096
#i32B

	full_text
	
i32 %25
4sextB,
*
	full_text

%27 = sext i32 %26 to i64
#i32B

	full_text
	
i32 %26
VgetelementptrBE
C
	full_text6
4
2%28 = getelementptr inbounds i16, i16* %0, i64 %27
#i64B

	full_text
	
i64 %27
2mulB+
)
	full_text

%29 = mul i32 %17, 17536
#i32B

	full_text
	
i32 %17
4sextB,
*
	full_text

%30 = sext i32 %29 to i64
#i32B

	full_text
	
i32 %29
WgetelementptrBF
D
	full_text7
5
3%31 = getelementptr inbounds i16, i16* %28, i64 %30
%i16*B

	full_text


i16* %28
#i64B

	full_text
	
i64 %30
/mulB(
&
	full_text

%32 = mul i32 %15, 17
#i32B

	full_text
	
i32 %15
-shlB&
$
	full_text

%33 = shl i64 %7, 2
"i64B

	full_text


i64 %7
6truncB-
+
	full_text

%34 = trunc i64 %33 to i32
#i64B

	full_text
	
i64 %33
.orB(
&
	full_text

%35 = or i32 %23, %34
#i32B

	full_text
	
i32 %23
#i32B

	full_text
	
i32 %34
4addB-
+
	full_text

%36 = add nsw i32 %32, %35
#i32B

	full_text
	
i32 %32
#i32B

	full_text
	
i32 %35
5mulB.
,
	full_text

%37 = mul nsw i32 %36, 1096
#i32B

	full_text
	
i32 %36
4sextB,
*
	full_text

%38 = sext i32 %37 to i64
#i32B

	full_text
	
i32 %37
VgetelementptrBE
C
	full_text6
4
2%39 = getelementptr inbounds i16, i16* %0, i64 %38
#i64B

	full_text
	
i64 %38
1mulB*
(
	full_text

%40 = mul i32 %17, 8768
#i32B

	full_text
	
i32 %17
4sextB,
*
	full_text

%41 = sext i32 %40 to i64
#i32B

	full_text
	
i32 %40
WgetelementptrBF
D
	full_text7
5
3%42 = getelementptr inbounds i16, i16* %39, i64 %41
%i16*B

	full_text


i16* %39
#i64B

	full_text
	
i64 %41
5icmpB-
+
	full_text

%43 = icmp slt i32 %8, 100
"i32B

	full_text


i32 %8
8brB2
0
	full_text#
!
br i1 %43, label %44, label %60
!i1B

	full_text


i1 %43
0mul8B'
%
	full_text

%45 = mul i32 %15, 9
%i328B

	full_text
	
i32 %15
/or8B'
%
	full_text

%46 = or i32 %6, %34
$i328B

	full_text


i32 %6
%i328B

	full_text
	
i32 %34
6add8B-
+
	full_text

%47 = add nsw i32 %45, %46
%i328B

	full_text
	
i32 %45
%i328B

	full_text
	
i32 %46
7mul8B.
,
	full_text

%48 = mul nsw i32 %47, 1096
%i328B

	full_text
	
i32 %47
6sext8B,
*
	full_text

%49 = sext i32 %48 to i64
%i328B

	full_text
	
i32 %48
Xgetelementptr8BE
C
	full_text6
4
2%50 = getelementptr inbounds i16, i16* %0, i64 %49
%i648B

	full_text
	
i64 %49
Ygetelementptr8BF
D
	full_text7
5
3%51 = getelementptr inbounds i16, i16* %50, i64 %41
'i16*8B

	full_text


i16* %50
%i648B

	full_text
	
i64 %41
0mul8B'
%
	full_text

%52 = mul i32 %15, 5
%i328B

	full_text
	
i32 %15
5add8B,
*
	full_text

%53 = add nsw i32 %52, %5
%i328B

	full_text
	
i32 %52
$i328B

	full_text


i32 %5
7mul8B.
,
	full_text

%54 = mul nsw i32 %53, 1096
%i328B

	full_text
	
i32 %53
6sext8B,
*
	full_text

%55 = sext i32 %54 to i64
%i328B

	full_text
	
i32 %54
Xgetelementptr8BE
C
	full_text6
4
2%56 = getelementptr inbounds i16, i16* %0, i64 %55
%i648B

	full_text
	
i64 %55
3mul8B*
(
	full_text

%57 = mul i32 %17, 4384
%i328B

	full_text
	
i32 %17
6sext8B,
*
	full_text

%58 = sext i32 %57 to i64
%i328B

	full_text
	
i32 %57
Ygetelementptr8BF
D
	full_text7
5
3%59 = getelementptr inbounds i16, i16* %56, i64 %58
'i16*8B

	full_text


i16* %56
%i648B

	full_text
	
i64 %58
'br8B

	full_text

br label %60
Dphi8B;
9
	full_text,
*
(%61 = phi i16* [ %59, %44 ], [ %19, %3 ]
'i16*8B

	full_text


i16* %59
'i16*8B

	full_text


i16* %19
Dphi8B;
9
	full_text,
*
(%62 = phi i16* [ %51, %44 ], [ %19, %3 ]
'i16*8B

	full_text


i16* %51
'i16*8B

	full_text


i16* %19
8icmp8B.
,
	full_text

%63 = icmp slt i32 %14, 545
%i328B

	full_text
	
i32 %14
:br8B2
0
	full_text#
!
br i1 %63, label %64, label %71
#i18B

	full_text


i1 %63
>bitcast8B1
/
	full_text"
 
%65 = bitcast i16* %31 to i32*
'i16*8B

	full_text


i16* %31
>bitcast8B1
/
	full_text"
 
%66 = bitcast i16* %42 to i32*
'i16*8B

	full_text


i16* %42
>bitcast8B1
/
	full_text"
 
%67 = bitcast i16* %62 to i32*
'i16*8B

	full_text


i16* %62
>bitcast8B1
/
	full_text"
 
%68 = bitcast i16* %61 to i32*
'i16*8B

	full_text


i16* %61
1shl8B(
&
	full_text

%69 = shl i64 %13, 32
%i648B

	full_text
	
i64 %13
9ashr8B/
-
	full_text 

%70 = ashr exact i64 %69, 32
%i648B

	full_text
	
i64 %69
'br8B

	full_text

br label %72
$ret8B

	full_text


ret void
Dphi8B;
9
	full_text,
*
(%73 = phi i64 [ %70, %64 ], [ %96, %72 ]
%i648B

	full_text
	
i64 %70
%i648B

	full_text
	
i64 %96
Ygetelementptr8BF
D
	full_text7
5
3%74 = getelementptr inbounds i32, i32* %65, i64 %73
'i32*8B

	full_text


i32* %65
%i648B

	full_text
	
i64 %73
Hload8B>
<
	full_text/
-
+%75 = load i32, i32* %74, align 4, !tbaa !8
'i32*8B

	full_text


i32* %74
6add8B-
+
	full_text

%76 = add nsw i64 %73, 548
%i648B

	full_text
	
i64 %73
Ygetelementptr8BF
D
	full_text7
5
3%77 = getelementptr inbounds i32, i32* %65, i64 %76
'i32*8B

	full_text


i32* %65
%i648B

	full_text
	
i64 %76
Hload8B>
<
	full_text/
-
+%78 = load i32, i32* %77, align 4, !tbaa !8
'i32*8B

	full_text


i32* %77
7add8B.
,
	full_text

%79 = add nsw i64 %73, 2192
%i648B

	full_text
	
i64 %73
Ygetelementptr8BF
D
	full_text7
5
3%80 = getelementptr inbounds i32, i32* %65, i64 %79
'i32*8B

	full_text


i32* %65
%i648B

	full_text
	
i64 %79
Hload8B>
<
	full_text/
-
+%81 = load i32, i32* %80, align 4, !tbaa !8
'i32*8B

	full_text


i32* %80
7add8B.
,
	full_text

%82 = add nsw i64 %73, 2740
%i648B

	full_text
	
i64 %73
Ygetelementptr8BF
D
	full_text7
5
3%83 = getelementptr inbounds i32, i32* %65, i64 %82
'i32*8B

	full_text


i32* %65
%i648B

	full_text
	
i64 %82
Hload8B>
<
	full_text/
-
+%84 = load i32, i32* %83, align 4, !tbaa !8
'i32*8B

	full_text


i32* %83
2add8B)
'
	full_text

%85 = add i32 %81, %75
%i328B

	full_text
	
i32 %81
%i328B

	full_text
	
i32 %75
Ygetelementptr8BF
D
	full_text7
5
3%86 = getelementptr inbounds i32, i32* %66, i64 %73
'i32*8B

	full_text


i32* %66
%i648B

	full_text
	
i64 %73
Hstore8B=
;
	full_text.
,
*store i32 %85, i32* %86, align 4, !tbaa !8
%i328B

	full_text
	
i32 %85
'i32*8B

	full_text


i32* %86
2add8B)
'
	full_text

%87 = add i32 %84, %78
%i328B

	full_text
	
i32 %84
%i328B

	full_text
	
i32 %78
Ygetelementptr8BF
D
	full_text7
5
3%88 = getelementptr inbounds i32, i32* %66, i64 %76
'i32*8B

	full_text


i32* %66
%i648B

	full_text
	
i64 %76
Hstore8B=
;
	full_text.
,
*store i32 %87, i32* %88, align 4, !tbaa !8
%i328B

	full_text
	
i32 %87
'i32*8B

	full_text


i32* %88
2add8B)
'
	full_text

%89 = add i32 %78, %75
%i328B

	full_text
	
i32 %78
%i328B

	full_text
	
i32 %75
Ygetelementptr8BF
D
	full_text7
5
3%90 = getelementptr inbounds i32, i32* %67, i64 %73
'i32*8B

	full_text


i32* %67
%i648B

	full_text
	
i64 %73
Hstore8B=
;
	full_text.
,
*store i32 %89, i32* %90, align 4, !tbaa !8
%i328B

	full_text
	
i32 %89
'i32*8B

	full_text


i32* %90
2add8B)
'
	full_text

%91 = add i32 %84, %81
%i328B

	full_text
	
i32 %84
%i328B

	full_text
	
i32 %81
7add8B.
,
	full_text

%92 = add nsw i64 %73, 1096
%i648B

	full_text
	
i64 %73
Ygetelementptr8BF
D
	full_text7
5
3%93 = getelementptr inbounds i32, i32* %67, i64 %92
'i32*8B

	full_text


i32* %67
%i648B

	full_text
	
i64 %92
Hstore8B=
;
	full_text.
,
*store i32 %91, i32* %93, align 4, !tbaa !8
%i328B

	full_text
	
i32 %91
'i32*8B

	full_text


i32* %93
2add8B)
'
	full_text

%94 = add i32 %91, %89
%i328B

	full_text
	
i32 %91
%i328B

	full_text
	
i32 %89
Ygetelementptr8BF
D
	full_text7
5
3%95 = getelementptr inbounds i32, i32* %68, i64 %73
'i32*8B

	full_text


i32* %68
%i648B

	full_text
	
i64 %73
Hstore8B=
;
	full_text.
,
*store i32 %94, i32* %95, align 4, !tbaa !8
%i328B

	full_text
	
i32 %94
'i32*8B

	full_text


i32* %95
5add8B,
*
	full_text

%96 = add nsw i64 %73, 32
%i648B

	full_text
	
i64 %73
8icmp8B.
,
	full_text

%97 = icmp slt i64 %73, 513
%i648B

	full_text
	
i64 %73
:br8B2
0
	full_text#
!
br i1 %97, label %72, label %71
#i18B

	full_text


i1 %97
$i328B

	full_text


i32 %1
$i328B

	full_text


i32 %2
&i16*8B

	full_text
	
i16* %0
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
#i328B

	full_text	

i32 1
&i648B

	full_text


i64 2192
$i328B

	full_text


i32 17
#i328B

	full_text	

i32 5
#i328B

	full_text	

i32 0
$i328B

	full_text


i32 25
%i648B

	full_text
	
i64 513
#i328B

	full_text	

i32 9
&i648B

	full_text


i64 2740
&i328B

	full_text


i32 8768
&i648B

	full_text


i64 1096
#i648B

	full_text	

i64 3
&i328B

	full_text


i32 1096
'i328B

	full_text

	i32 17536
#i648B

	full_text	

i64 2
%i328B

	full_text
	
i32 100
&i328B

	full_text


i32 4384
%i328B

	full_text
	
i32 545
%i648B

	full_text
	
i64 548
#i648B

	full_text	

i64 1
$i648B

	full_text


i64 32       	  

                        !" !! #$ ## %& %' %% () (* (( +, ++ -. -- /0 // 12 11 34 33 56 57 55 89 88 :; :: <= << >? >@ >> AB AC AA DE DD FG FF HI HH JK JJ LM LL NO NP NN QR QQ ST SV UU WX WY WW Z[ Z\ ZZ ]^ ]] _` __ ab aa cd ce cc fg ff hi hj hh kl kk mn mm op oo qr qq st ss uv uw uu xz y{ yy |} |~ || Ä  ÅÇ ÅÑ ÉÉ ÖÜ ÖÖ áà áá âä ââ ãå ãã çé çç èí ë
ì ëë îï î
ñ îî óò óó ôö ôô õú õ
ù õõ ûü ûû †° †† ¢£ ¢
§ ¢¢ •¶ •• ß® ßß ©™ ©
´ ©© ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±
≥ ±± ¥µ ¥
∂ ¥¥ ∑∏ ∑
π ∑∑ ∫ª ∫
º ∫∫ Ωæ Ω
ø ΩΩ ¿¡ ¿
¬ ¿¿ √ƒ √
≈ √√ ∆« ∆
» ∆∆ …  …
À …… ÃÕ ÃÃ Œœ Œ
– ŒŒ —“ —
” —— ‘’ ‘
÷ ‘‘ ◊ÿ ◊
Ÿ ◊◊ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›› ﬂ‡ ﬂﬂ ·‚ ·„ 	„ 	‰ Â /Â HÂ aÂ o    	
            " $# &! ' )% *( ,+ .- 0 21 4/ 63 7 9 ;: =# ?< @8 B> CA ED GF I KJ MH OL P RQ T V X< YU [W \Z ^] `_ ba dL e gf i jh lk nm p rq to vs wu z {c } ~ Ä Ç5 ÑN Ü| ày ä åã éç í› ìÉ ïë ñî òë öÉ úô ùõ üë °É £† §¢ ¶ë ®É ™ß ´© ≠• Øó ∞Ö ≤ë ≥Æ µ± ∂¨ ∏û πÖ ªô º∑ æ∫ øû ¡ó ¬á ƒë ≈¿ «√ »¨  • Àë Õá œÃ –… “Œ ”… ’¿ ÷â ÿë Ÿ‘ €◊ ‹ë ﬁë ‡ﬂ ‚S US yx yÅ ÉÅ êè ë· ë· ê ê ÊÊ ËË ÁÁ ÊÊ  ËË  ÁÁ  ÊÊ 
 ÁÁ 
 ËË  ËË È 	È È 	È #
Í †	Î 8	Ï fÌ 
Ì 	Ó 
Ô ﬂ	 U
Ò ß	Ú J
Û Ã	Ù 	ı +	ı D	ı ]	ı k	ˆ 1	˜ :	¯ Q	˘ q	˙ 
˚ ô	¸ 
˝ ã
˝ ç
˝ ›"
larger_sad_calc_8"
_Z12get_local_idj"
_Z12get_group_idj"

_Z5mul24ii*ù
$parboil-0.2-sad-larger_sad_calc_8.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Å
 
transfer_bytes_log1p
˙ìÖA

devmap_label
 

wgsize
Ä

wgsize_log1p
˙ìÖA

transfer_bytes
‡ò¡