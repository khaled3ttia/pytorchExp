

[external]
AbitcastB6
4
	full_text'
%
#%8 = bitcast i32* %2 to [12 x i32]*
KcallBC
A
	full_text4
2
0%9 = tail call i64 @_Z13get_global_idj(i32 0) #2
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
NcallBF
D
	full_text7
5
3%11 = tail call i64 @_Z15get_global_sizej(i32 0) #2
6truncB-
+
	full_text

%12 = trunc i64 %11 to i32
#i64B

	full_text
	
i64 %11
.shlB'
%
	full_text

%13 = shl i64 %9, 32
"i64B

	full_text


i64 %9
7ashrB/
-
	full_text 

%14 = ashr exact i64 %13, 32
#i64B

	full_text
	
i64 %13
VgetelementptrBE
C
	full_text6
4
2%15 = getelementptr inbounds i32, i32* %4, i64 %14
#i64B

	full_text
	
i64 %14
FloadB>
<
	full_text/
-
+%16 = load i32, i32* %15, align 4, !tbaa !8
%i32*B

	full_text


i32* %15
5icmpB-
+
	full_text

%17 = icmp slt i32 %16, %6
#i32B

	full_text
	
i32 %16
9brB3
1
	full_text$
"
 br i1 %17, label %18, label %113
!i1B

	full_text


i1 %17
Xgetelementptr8BE
C
	full_text6
4
2%19 = getelementptr inbounds i32, i32* %5, i64 %14
%i648B

	full_text
	
i64 %14
Hload8B>
<
	full_text/
-
+%20 = load i32, i32* %19, align 4, !tbaa !8
'i32*8B

	full_text


i32* %19
4add8B+
)
	full_text

%21 = add nsw i32 %16, 1
%i328B

	full_text
	
i32 %16
5icmp8B+
)
	full_text

%22 = icmp sgt i32 %6, 0
:br8B2
0
	full_text#
!
br i1 %22, label %23, label %54
#i18B

	full_text


i1 %22
5zext8B+
)
	full_text

%24 = zext i32 %6 to i64
'br8B

	full_text

br label %25
Bphi8B9
7
	full_text*
(
&%26 = phi i64 [ 0, %23 ], [ %52, %51 ]
%i648B

	full_text
	
i64 %52
Xgetelementptr8BE
C
	full_text6
4
2%27 = getelementptr inbounds i32, i32* %1, i64 %26
%i648B

	full_text
	
i64 %26
Hload8B>
<
	full_text/
-
+%28 = load i32, i32* %27, align 4, !tbaa !8
'i32*8B

	full_text


i32* %27
6icmp8B,
*
	full_text

%29 = icmp sgt i32 %28, 0
%i328B

	full_text
	
i32 %28
:br8B2
0
	full_text#
!
br i1 %29, label %30, label %51
#i18B

	full_text


i1 %29
'br8B

	full_text

br label %31
Bphi8B9
7
	full_text*
(
&%32 = phi i64 [ %48, %46 ], [ 0, %30 ]
%i648B

	full_text
	
i64 %48
Dphi8B;
9
	full_text,
*
(%33 = phi i32 [ %47, %46 ], [ %28, %30 ]
%i328B

	full_text
	
i32 %47
%i328B

	full_text
	
i32 %28
ogetelementptr8B\
Z
	full_textM
K
I%34 = getelementptr inbounds [12 x i32], [12 x i32]* %8, i64 %26, i64 %32
4[12 x i32]*8B!

	full_text

[12 x i32]* %8
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %32
Hload8B>
<
	full_text/
-
+%35 = load i32, i32* %34, align 4, !tbaa !8
'i32*8B

	full_text


i32* %34
8icmp8B.
,
	full_text

%36 = icmp sge i32 %35, %16
%i328B

	full_text
	
i32 %35
%i328B

	full_text
	
i32 %16
8icmp8B.
,
	full_text

%37 = icmp slt i32 %35, %20
%i328B

	full_text
	
i32 %35
%i328B

	full_text
	
i32 %20
1and8B(
&
	full_text

%38 = and i1 %36, %37
#i18B

	full_text


i1 %36
#i18B

	full_text


i1 %37
:br8B2
0
	full_text#
!
br i1 %38, label %39, label %46
#i18B

	full_text


i1 %38
4add8B+
)
	full_text

%40 = add nsw i32 %35, 1
%i328B

	full_text
	
i32 %35
6sext8B,
*
	full_text

%41 = sext i32 %40 to i64
%i328B

	full_text
	
i32 %40
Xgetelementptr8BE
C
	full_text6
4
2%42 = getelementptr inbounds i32, i32* %0, i64 %41
%i648B

	full_text
	
i64 %41
Hload8B>
<
	full_text/
-
+%43 = load i32, i32* %42, align 4, !tbaa !8
'i32*8B

	full_text


i32* %42
6add8B-
+
	full_text

%44 = add nsw i32 %43, %33
%i328B

	full_text
	
i32 %43
%i328B

	full_text
	
i32 %33
Hstore8B=
;
	full_text.
,
*store i32 %44, i32* %42, align 4, !tbaa !8
%i328B

	full_text
	
i32 %44
'i32*8B

	full_text


i32* %42
Hload8B>
<
	full_text/
-
+%45 = load i32, i32* %27, align 4, !tbaa !8
'i32*8B

	full_text


i32* %27
'br8B

	full_text

br label %46
Dphi8B;
9
	full_text,
*
(%47 = phi i32 [ %33, %31 ], [ %45, %39 ]
%i328B

	full_text
	
i32 %33
%i328B

	full_text
	
i32 %45
8add8B/
-
	full_text 

%48 = add nuw nsw i64 %32, 1
%i648B

	full_text
	
i64 %32
6sext8B,
*
	full_text

%49 = sext i32 %47 to i64
%i328B

	full_text
	
i32 %47
8icmp8B.
,
	full_text

%50 = icmp slt i64 %48, %49
%i648B

	full_text
	
i64 %48
%i648B

	full_text
	
i64 %49
:br8B2
0
	full_text#
!
br i1 %50, label %31, label %51
#i18B

	full_text


i1 %50
8add8B/
-
	full_text 

%52 = add nuw nsw i64 %26, 1
%i648B

	full_text
	
i64 %26
7icmp8B-
+
	full_text

%53 = icmp eq i64 %52, %24
%i648B

	full_text
	
i64 %52
%i648B

	full_text
	
i64 %24
:br8B2
0
	full_text#
!
br i1 %53, label %54, label %25
#i18B

	full_text


i1 %53
5icmp8	B+
)
	full_text

%55 = icmp eq i32 %10, 0
%i328	B

	full_text
	
i32 %10
:br8	B2
0
	full_text#
!
br i1 %55, label %56, label %57
#i18	B

	full_text


i1 %55
Estore8
B:
8
	full_text+
)
'store i32 0, i32* %0, align 4, !tbaa !8
'br8
B

	full_text

br label %57
Bphi8B9
7
	full_text*
(
&%58 = phi i32 [ 0, %56 ], [ %21, %54 ]
%i328B

	full_text
	
i32 %21
8icmp8B.
,
	full_text

%59 = icmp slt i32 %58, %20
%i328B

	full_text
	
i32 %58
%i328B

	full_text
	
i32 %20
;br8B3
1
	full_text$
"
 br i1 %59, label %60, label %106
#i18B

	full_text


i1 %59
6sext8B,
*
	full_text

%61 = sext i32 %58 to i64
%i328B

	full_text
	
i32 %58
Xgetelementptr8BE
C
	full_text6
4
2%62 = getelementptr inbounds i32, i32* %0, i64 %61
%i648B

	full_text
	
i64 %61
Hload8B>
<
	full_text/
-
+%63 = load i32, i32* %62, align 4, !tbaa !8
'i32*8B

	full_text


i32* %62
6sext8B,
*
	full_text

%64 = sext i32 %20 to i64
%i328B

	full_text
	
i32 %20
6sub8B-
+
	full_text

%65 = sub nsw i64 %64, %61
%i648B

	full_text
	
i64 %64
%i648B

	full_text
	
i64 %61
5add8B,
*
	full_text

%66 = add nsw i64 %64, -1
%i648B

	full_text
	
i64 %64
6sub8B-
+
	full_text

%67 = sub nsw i64 %66, %61
%i648B

	full_text
	
i64 %66
%i648B

	full_text
	
i64 %61
0and8B'
%
	full_text

%68 = and i64 %65, 3
%i648B

	full_text
	
i64 %65
5icmp8B+
)
	full_text

%69 = icmp eq i64 %68, 0
%i648B

	full_text
	
i64 %68
:br8B2
0
	full_text#
!
br i1 %69, label %81, label %70
#i18B

	full_text


i1 %69
'br8B

	full_text

br label %71
Dphi8B;
9
	full_text,
*
(%72 = phi i32 [ %63, %70 ], [ %78, %71 ]
%i328B

	full_text
	
i32 %63
%i328B

	full_text
	
i32 %78
Dphi8B;
9
	full_text,
*
(%73 = phi i64 [ %61, %70 ], [ %75, %71 ]
%i648B

	full_text
	
i64 %61
%i648B

	full_text
	
i64 %75
Dphi8B;
9
	full_text,
*
(%74 = phi i64 [ %68, %70 ], [ %79, %71 ]
%i648B

	full_text
	
i64 %68
%i648B

	full_text
	
i64 %79
4add8B+
)
	full_text

%75 = add nsw i64 %73, 1
%i648B

	full_text
	
i64 %73
Xgetelementptr8BE
C
	full_text6
4
2%76 = getelementptr inbounds i32, i32* %0, i64 %75
%i648B

	full_text
	
i64 %75
Hload8B>
<
	full_text/
-
+%77 = load i32, i32* %76, align 4, !tbaa !8
'i32*8B

	full_text


i32* %76
6add8B-
+
	full_text

%78 = add nsw i32 %72, %77
%i328B

	full_text
	
i32 %72
%i328B

	full_text
	
i32 %77
Hstore8B=
;
	full_text.
,
*store i32 %78, i32* %76, align 4, !tbaa !8
%i328B

	full_text
	
i32 %78
'i32*8B

	full_text


i32* %76
1add8B(
&
	full_text

%79 = add i64 %74, -1
%i648B

	full_text
	
i64 %74
5icmp8B+
)
	full_text

%80 = icmp eq i64 %79, 0
%i648B

	full_text
	
i64 %79
Jbr8BB
@
	full_text3
1
/br i1 %80, label %81, label %71, !llvm.loop !12
#i18B

	full_text


i1 %80
Dphi8B;
9
	full_text,
*
(%82 = phi i32 [ %63, %60 ], [ %78, %71 ]
%i328B

	full_text
	
i32 %63
%i328B

	full_text
	
i32 %78
Dphi8B;
9
	full_text,
*
(%83 = phi i64 [ %61, %60 ], [ %75, %71 ]
%i648B

	full_text
	
i64 %61
%i648B

	full_text
	
i64 %75
6icmp8B,
*
	full_text

%84 = icmp ult i64 %67, 3
%i648B

	full_text
	
i64 %67
;br8B3
1
	full_text$
"
 br i1 %84, label %106, label %85
#i18B

	full_text


i1 %84
'br8B

	full_text

br label %86
Ephi8B<
:
	full_text-
+
)%87 = phi i32 [ %82, %85 ], [ %104, %86 ]
%i328B

	full_text
	
i32 %82
&i328B

	full_text


i32 %104
Ephi8B<
:
	full_text-
+
)%88 = phi i64 [ %83, %85 ], [ %101, %86 ]
%i648B

	full_text
	
i64 %83
&i648B

	full_text


i64 %101
4add8B+
)
	full_text

%89 = add nsw i64 %88, 1
%i648B

	full_text
	
i64 %88
Xgetelementptr8BE
C
	full_text6
4
2%90 = getelementptr inbounds i32, i32* %0, i64 %89
%i648B

	full_text
	
i64 %89
Hload8B>
<
	full_text/
-
+%91 = load i32, i32* %90, align 4, !tbaa !8
'i32*8B

	full_text


i32* %90
6add8B-
+
	full_text

%92 = add nsw i32 %87, %91
%i328B

	full_text
	
i32 %87
%i328B

	full_text
	
i32 %91
Hstore8B=
;
	full_text.
,
*store i32 %92, i32* %90, align 4, !tbaa !8
%i328B

	full_text
	
i32 %92
'i32*8B

	full_text


i32* %90
4add8B+
)
	full_text

%93 = add nsw i64 %88, 2
%i648B

	full_text
	
i64 %88
Xgetelementptr8BE
C
	full_text6
4
2%94 = getelementptr inbounds i32, i32* %0, i64 %93
%i648B

	full_text
	
i64 %93
Hload8B>
<
	full_text/
-
+%95 = load i32, i32* %94, align 4, !tbaa !8
'i32*8B

	full_text


i32* %94
6add8B-
+
	full_text

%96 = add nsw i32 %92, %95
%i328B

	full_text
	
i32 %92
%i328B

	full_text
	
i32 %95
Hstore8B=
;
	full_text.
,
*store i32 %96, i32* %94, align 4, !tbaa !8
%i328B

	full_text
	
i32 %96
'i32*8B

	full_text


i32* %94
4add8B+
)
	full_text

%97 = add nsw i64 %88, 3
%i648B

	full_text
	
i64 %88
Xgetelementptr8BE
C
	full_text6
4
2%98 = getelementptr inbounds i32, i32* %0, i64 %97
%i648B

	full_text
	
i64 %97
Hload8B>
<
	full_text/
-
+%99 = load i32, i32* %98, align 4, !tbaa !8
'i32*8B

	full_text


i32* %98
7add8B.
,
	full_text

%100 = add nsw i32 %96, %99
%i328B

	full_text
	
i32 %96
%i328B

	full_text
	
i32 %99
Istore8B>
<
	full_text/
-
+store i32 %100, i32* %98, align 4, !tbaa !8
&i328B

	full_text


i32 %100
'i32*8B

	full_text


i32* %98
5add8B,
*
	full_text

%101 = add nsw i64 %88, 4
%i648B

	full_text
	
i64 %88
Zgetelementptr8BG
E
	full_text8
6
4%102 = getelementptr inbounds i32, i32* %0, i64 %101
&i648B

	full_text


i64 %101
Jload8B@
>
	full_text1
/
-%103 = load i32, i32* %102, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %102
9add8B0
.
	full_text!

%104 = add nsw i32 %100, %103
&i328B

	full_text


i32 %100
&i328B

	full_text


i32 %103
Jstore8B?
=
	full_text0
.
,store i32 %104, i32* %102, align 4, !tbaa !8
&i328B

	full_text


i32 %104
(i32*8B

	full_text

	i32* %102
9icmp8B/
-
	full_text 

%105 = icmp eq i64 %101, %64
&i648B

	full_text


i64 %101
%i648B

	full_text
	
i64 %64
<br8B4
2
	full_text%
#
!br i1 %105, label %106, label %86
$i18B

	full_text
	
i1 %105
9icmp8B/
-
	full_text 

%107 = icmp slt i32 %10, %12
%i328B

	full_text
	
i32 %10
%i328B

	full_text
	
i32 %12
=br8B5
3
	full_text&
$
"br i1 %107, label %108, label %113
$i18B

	full_text
	
i1 %107
7sext8B-
+
	full_text

%109 = sext i32 %20 to i64
%i328B

	full_text
	
i32 %20
Zgetelementptr8BG
E
	full_text8
6
4%110 = getelementptr inbounds i32, i32* %0, i64 %109
&i648B

	full_text


i64 %109
Jload8B@
>
	full_text1
/
-%111 = load i32, i32* %110, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %110
Ygetelementptr8BF
D
	full_text7
5
3%112 = getelementptr inbounds i32, i32* %3, i64 %14
%i648B

	full_text
	
i64 %14
Jstore8B?
=
	full_text0
.
,store i32 %111, i32* %112, align 4, !tbaa !8
&i328B

	full_text


i32 %111
(i32*8B

	full_text

	i32* %112
(br8B 

	full_text

br label %113
$ret8B

	full_text


ret void
&i32*8B

	full_text
	
i32* %2
&i32*8B

	full_text
	
i32* %4
&i32*8B

	full_text
	
i32* %5
$i328B

	full_text


i32 %6
&i32*8B

	full_text
	
i32* %1
&i32*8B

	full_text
	
i32* %0
&i32*8B

	full_text
	
i32* %3
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
#i648B

	full_text	

i64 3
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 1
$i648B

	full_text


i64 -1
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 4
#i648B

	full_text	

i64 2
#i328B

	full_text	

i32 1
$i648B

	full_text


i64 32       	  
 

                    !" !! #$ ## %& %% '( '+ ** ,- ,. ,, /0 /1 /2 // 34 33 56 57 55 89 8: 88 ;< ;= ;; >? >A @@ BC BB DE DD FG FF HI HJ HH KL KM KK NO NN PR QS QQ TU TT VW VV XY XZ XX [\ [^ ]] _` _a __ bc be dd fg fh ik jj lm ln ll op or qq st ss uv uu wx ww yz y{ yy |} || ~ ~	Ä ~~ ÅÇ ÅÅ ÉÑ ÉÉ ÖÜ Öâ à
ä àà ãå ã
ç ãã éè é
ê éé ëí ëë ì
î ìì ïñ ïï óò ó
ô óó öõ ö
ú öö ùû ùù ü† üü °¢ °§ £
• ££ ¶ß ¶
® ¶¶ ©™ ©© ´¨ ´Ø Æ
∞ ÆÆ ±≤ ±
≥ ±± ¥µ ¥¥ ∂
∑ ∂∂ ∏π ∏∏ ∫ª ∫
º ∫∫ Ωæ Ω
ø ΩΩ ¿¡ ¿¿ ¬
√ ¬¬ ƒ≈ ƒƒ ∆« ∆
» ∆∆ …  …
À …… ÃÕ ÃÃ Œ
œ ŒŒ –— –– “” “
‘ ““ ’÷ ’
◊ ’’ ÿŸ ÿÿ ⁄
€ ⁄⁄ ‹› ‹‹ ﬁﬂ ﬁ
‡ ﬁﬁ ·‚ ·
„ ·· ‰Â ‰
Ê ‰‰ ÁË ÁÍ È
Î ÈÈ ÏÌ ÏÔ ÓÓ 
Ò  ÚÛ ÚÚ Ù
ı ÙÙ ˆ˜ ˆ
¯ ˆˆ ˘˚ ¸ ˝ 	˛ ˛ ˛ ˇ !Ä D	Ä hÄ sÄ ìÄ ∂Ä ¬Ä ŒÄ ⁄Ä Å Ù   	 
    
    ]   "! $# &% (T +Q -# . 0 1* 2/ 43 6 73 9 :5 <8 =; ?3 A@ CB ED GF I, JH LD M! O, RN S* UQ WT YV ZX \ ^] ` a_ c ed g kj m nl pj rq ts v xw zq {w }| q Äy ÇÅ ÑÉ Üu âó äq åë çÅ èù êã íë îì ñà òï ôó õì úé ûù †ü ¢u §ó •q ßë ®~ ™© ¨£ Øﬁ ∞¶ ≤ÿ ≥± µ¥ ∑∂ πÆ ª∏ º∫ æ∂ ø± ¡¿ √¬ ≈∫ «ƒ »∆  ¬ À± ÕÃ œŒ —∆ ”– ‘“ ÷Œ ◊± Ÿÿ €⁄ ›“ ﬂ‹ ‡ﬁ ‚⁄ „ÿ Âw Ê‰ Ë Í ÎÈ Ì ÔÓ Ò Û
 ıÚ ˜Ù ¯  ˙  d f hf j' )' ]i jo qo È) *b db Ö £Ö áÏ ÓÏ ˙> @> Q´ È´ ≠á à˘ ˙P Q[ *[ ]≠ Æ° £° àÁ ÈÁ Æ ˙ ÇÇ ÉÉ ÇÇ  ÉÉ 
Ñ Å
Ñ ©
Ñ ÃÖ Ö 	Ö 	Ö %	Ö dÖ hÖ j	Ü T	Ü ]
Ü ë
Ü ¥	á |
á ùà 	à *
à É
à ü
â ÿ
ä ¿	ã 	ã @	å 	å 
"	
makea_1"
_Z13get_global_idj"
_Z15get_global_sizej*ä
npb-CG-makea_1.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Ä

devmap_label

 
transfer_bytes_log1p
Q éA

wgsize
 

wgsize_log1p
Q éA

transfer_bytes
‹…±