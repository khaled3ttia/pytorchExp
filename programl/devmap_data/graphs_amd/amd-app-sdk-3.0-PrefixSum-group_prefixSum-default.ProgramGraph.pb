

[external]
JcallBB
@
	full_text3
1
/%6 = tail call i64 @_Z12get_local_idj(i32 0) #3
4truncB+
)
	full_text

%7 = trunc i64 %6 to i32
"i64B

	full_text


i64 %6
LcallBD
B
	full_text5
3
1%8 = tail call i64 @_Z14get_local_sizej(i32 0) #3
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
KcallBC
A
	full_text4
2
0%10 = tail call i64 @_Z12get_group_idj(i32 0) #3
6truncB-
+
	full_text

%11 = trunc i64 %10 to i32
#i64B

	full_text
	
i64 %10
3mulB,
*
	full_text

%12 = mul nsw i32 %11, %9
#i32B

	full_text
	
i32 %11
"i32B

	full_text


i32 %9
3addB,
*
	full_text

%13 = add nsw i32 %12, %7
#i32B

	full_text
	
i32 %12
"i32B

	full_text


i32 %7
2shlB+
)
	full_text

%14 = shl nsw i32 %13, 1
#i32B

	full_text
	
i32 %13
,orB&
$
	full_text

%15 = or i32 %14, 1
#i32B

	full_text
	
i32 %14
/mulB(
&
	full_text

%16 = mul i32 %15, %4
#i32B

	full_text
	
i32 %15
/addB(
&
	full_text

%17 = add i32 %16, -1
#i32B

	full_text
	
i32 %16
5icmpB-
+
	full_text

%18 = icmp ult i32 %17, %3
#i32B

	full_text
	
i32 %17
8brB2
0
	full_text#
!
br i1 %18, label %19, label %28
!i1B

	full_text


i1 %18
6sext8B,
*
	full_text

%20 = sext i32 %17 to i64
%i328B

	full_text
	
i32 %17
\getelementptr8BI
G
	full_text:
8
6%21 = getelementptr inbounds float, float* %1, i64 %20
%i648B

	full_text
	
i64 %20
@bitcast8B3
1
	full_text$
"
 %22 = bitcast float* %21 to i32*
+float*8B

	full_text


float* %21
Hload8B>
<
	full_text/
-
+%23 = load i32, i32* %22, align 4, !tbaa !8
'i32*8B

	full_text


i32* %22
3shl8B*
(
	full_text

%24 = shl nsw i32 %7, 1
$i328B

	full_text


i32 %7
6sext8B,
*
	full_text

%25 = sext i32 %24 to i64
%i328B

	full_text
	
i32 %24
\getelementptr8BI
G
	full_text:
8
6%26 = getelementptr inbounds float, float* %2, i64 %25
%i648B

	full_text
	
i64 %25
@bitcast8B3
1
	full_text$
"
 %27 = bitcast float* %26 to i32*
+float*8B

	full_text


float* %26
Hstore8B=
;
	full_text.
,
*store i32 %23, i32* %27, align 4, !tbaa !8
%i328B

	full_text
	
i32 %23
'i32*8B

	full_text


i32* %27
'br8B

	full_text

br label %28
1add8B(
&
	full_text

%29 = add i32 %17, %4
%i328B

	full_text
	
i32 %17
7icmp8B-
+
	full_text

%30 = icmp ult i32 %29, %3
%i328B

	full_text
	
i32 %29
:br8B2
0
	full_text#
!
br i1 %30, label %31, label %41
#i18B

	full_text


i1 %30
6zext8B,
*
	full_text

%32 = zext i32 %29 to i64
%i328B

	full_text
	
i32 %29
\getelementptr8BI
G
	full_text:
8
6%33 = getelementptr inbounds float, float* %1, i64 %32
%i648B

	full_text
	
i64 %32
@bitcast8B3
1
	full_text$
"
 %34 = bitcast float* %33 to i32*
+float*8B

	full_text


float* %33
Hload8B>
<
	full_text/
-
+%35 = load i32, i32* %34, align 4, !tbaa !8
'i32*8B

	full_text


i32* %34
3shl8B*
(
	full_text

%36 = shl nsw i32 %7, 1
$i328B

	full_text


i32 %7
.or8B&
$
	full_text

%37 = or i32 %36, 1
%i328B

	full_text
	
i32 %36
6sext8B,
*
	full_text

%38 = sext i32 %37 to i64
%i328B

	full_text
	
i32 %37
\getelementptr8BI
G
	full_text:
8
6%39 = getelementptr inbounds float, float* %2, i64 %38
%i648B

	full_text
	
i64 %38
@bitcast8B3
1
	full_text$
"
 %40 = bitcast float* %39 to i32*
+float*8B

	full_text


float* %39
Hstore8B=
;
	full_text.
,
*store i32 %35, i32* %40, align 4, !tbaa !8
%i328B

	full_text
	
i32 %35
'i32*8B

	full_text


i32* %40
'br8B

	full_text

br label %41
1lshr8B'
%
	full_text

%42 = lshr i32 %3, 1
5icmp8B+
)
	full_text

%43 = icmp eq i32 %42, 0
%i328B

	full_text
	
i32 %42
:br8B2
0
	full_text#
!
br i1 %43, label %48, label %44
#i18B

	full_text


i1 %43
3shl8B*
(
	full_text

%45 = shl nsw i32 %7, 1
$i328B

	full_text


i32 %7
.or8B&
$
	full_text

%46 = or i32 %45, 1
%i328B

	full_text
	
i32 %45
4add8B+
)
	full_text

%47 = add nsw i32 %45, 2
%i328B

	full_text
	
i32 %45
'br8B

	full_text

br label %51
Bphi8B9
7
	full_text*
(
&%49 = phi i32 [ 1, %41 ], [ %68, %67 ]
%i328B

	full_text
	
i32 %68
5icmp8B+
)
	full_text

%50 = icmp ugt i32 %3, 2
;br8B3
1
	full_text$
"
 br i1 %50, label %71, label %100
#i18B

	full_text


i1 %50
Dphi8B;
9
	full_text,
*
(%52 = phi i32 [ %42, %44 ], [ %69, %67 ]
%i328B

	full_text
	
i32 %42
%i328B

	full_text
	
i32 %69
Bphi8B9
7
	full_text*
(
&%53 = phi i32 [ 1, %44 ], [ %68, %67 ]
%i328B

	full_text
	
i32 %68
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
7icmp8B-
+
	full_text

%54 = icmp sgt i32 %52, %7
%i328B

	full_text
	
i32 %52
$i328B

	full_text


i32 %7
:br8B2
0
	full_text#
!
br i1 %54, label %55, label %67
#i18B

	full_text


i1 %54
6mul8B-
+
	full_text

%56 = mul nsw i32 %53, %46
%i328B

	full_text
	
i32 %53
%i328B

	full_text
	
i32 %46
5add8B,
*
	full_text

%57 = add nsw i32 %56, -1
%i328B

	full_text
	
i32 %56
6mul8B-
+
	full_text

%58 = mul nsw i32 %53, %47
%i328B

	full_text
	
i32 %53
%i328B

	full_text
	
i32 %47
5add8B,
*
	full_text

%59 = add nsw i32 %58, -1
%i328B

	full_text
	
i32 %58
6sext8B,
*
	full_text

%60 = sext i32 %57 to i64
%i328B

	full_text
	
i32 %57
\getelementptr8BI
G
	full_text:
8
6%61 = getelementptr inbounds float, float* %2, i64 %60
%i648B

	full_text
	
i64 %60
Lload8BB
@
	full_text3
1
/%62 = load float, float* %61, align 4, !tbaa !8
+float*8B

	full_text


float* %61
6sext8B,
*
	full_text

%63 = sext i32 %59 to i64
%i328B

	full_text
	
i32 %59
\getelementptr8BI
G
	full_text:
8
6%64 = getelementptr inbounds float, float* %2, i64 %63
%i648B

	full_text
	
i64 %63
Lload8BB
@
	full_text3
1
/%65 = load float, float* %64, align 4, !tbaa !8
+float*8B

	full_text


float* %64
6fadd8B,
*
	full_text

%66 = fadd float %62, %65
)float8B

	full_text

	float %62
)float8B

	full_text

	float %65
Lstore8BA
?
	full_text2
0
.store float %66, float* %64, align 4, !tbaa !8
)float8B

	full_text

	float %66
+float*8B

	full_text


float* %64
'br8B

	full_text

br label %67
0shl8	B'
%
	full_text

%68 = shl i32 %53, 1
%i328	B

	full_text
	
i32 %53
2lshr8	B(
&
	full_text

%69 = lshr i32 %52, 1
%i328	B

	full_text
	
i32 %52
5icmp8	B+
)
	full_text

%70 = icmp eq i32 %69, 0
%i328	B

	full_text
	
i32 %69
:br8	B2
0
	full_text#
!
br i1 %70, label %48, label %51
#i18	B

	full_text


i1 %70
7icmp8
B-
+
	full_text

%72 = icmp ult i32 %49, %3
%i328
B

	full_text
	
i32 %49
5zext8
B+
)
	full_text

%73 = zext i1 %72 to i32
#i18
B

	full_text


i1 %72
2shl8
B)
'
	full_text

%74 = shl i32 %49, %73
%i328
B

	full_text
	
i32 %49
%i328
B

	full_text
	
i32 %73
2ashr8
B(
&
	full_text

%75 = ashr i32 %74, 1
%i328
B

	full_text
	
i32 %74
6icmp8
B,
*
	full_text

%76 = icmp sgt i32 %75, 0
%i328
B

	full_text
	
i32 %75
;br8
B3
1
	full_text$
"
 br i1 %76, label %77, label %100
#i18
B

	full_text


i1 %76
3add8B*
(
	full_text

%78 = add nsw i32 %7, 1
$i328B

	full_text


i32 %7
'br8B

	full_text

br label %79
Dphi8B;
9
	full_text,
*
(%80 = phi i32 [ %74, %77 ], [ %83, %97 ]
%i328B

	full_text
	
i32 %74
%i328B

	full_text
	
i32 %83
Bphi8B9
7
	full_text*
(
&%81 = phi i32 [ 0, %77 ], [ %98, %97 ]
%i328B

	full_text
	
i32 %98
.or8B&
$
	full_text

%82 = or i32 %81, 1
%i328B

	full_text
	
i32 %81
2ashr8B(
&
	full_text

%83 = ashr i32 %80, 1
%i328B

	full_text
	
i32 %80
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
7icmp8B-
+
	full_text

%84 = icmp sgt i32 %82, %7
%i328B

	full_text
	
i32 %82
$i328B

	full_text


i32 %7
:br8B2
0
	full_text#
!
br i1 %84, label %85, label %97
#i18B

	full_text


i1 %84
6mul8B-
+
	full_text

%86 = mul nsw i32 %83, %78
%i328B

	full_text
	
i32 %83
%i328B

	full_text
	
i32 %78
5add8B,
*
	full_text

%87 = add nsw i32 %86, -1
%i328B

	full_text
	
i32 %86
2ashr8B(
&
	full_text

%88 = ashr i32 %80, 2
%i328B

	full_text
	
i32 %80
6add8B-
+
	full_text

%89 = add nsw i32 %87, %88
%i328B

	full_text
	
i32 %87
%i328B

	full_text
	
i32 %88
6sext8B,
*
	full_text

%90 = sext i32 %87 to i64
%i328B

	full_text
	
i32 %87
\getelementptr8BI
G
	full_text:
8
6%91 = getelementptr inbounds float, float* %2, i64 %90
%i648B

	full_text
	
i64 %90
Lload8BB
@
	full_text3
1
/%92 = load float, float* %91, align 4, !tbaa !8
+float*8B

	full_text


float* %91
6sext8B,
*
	full_text

%93 = sext i32 %89 to i64
%i328B

	full_text
	
i32 %89
\getelementptr8BI
G
	full_text:
8
6%94 = getelementptr inbounds float, float* %2, i64 %93
%i648B

	full_text
	
i64 %93
Lload8BB
@
	full_text3
1
/%95 = load float, float* %94, align 4, !tbaa !8
+float*8B

	full_text


float* %94
6fadd8B,
*
	full_text

%96 = fadd float %92, %95
)float8B

	full_text

	float %92
)float8B

	full_text

	float %95
Lstore8BA
?
	full_text2
0
.store float %96, float* %94, align 4, !tbaa !8
)float8B

	full_text

	float %96
+float*8B

	full_text


float* %94
'br8B

	full_text

br label %97
0shl8B'
%
	full_text

%98 = shl i32 %82, 1
%i328B

	full_text
	
i32 %82
8icmp8B.
,
	full_text

%99 = icmp slt i32 %98, %75
%i328B

	full_text
	
i32 %98
%i328B

	full_text
	
i32 %75
;br8B3
1
	full_text$
"
 br i1 %99, label %79, label %100
#i18B

	full_text


i1 %99
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
<br8B4
2
	full_text%
#
!br i1 %18, label %101, label %110
#i18B

	full_text


i1 %18
4shl8B+
)
	full_text

%102 = shl nsw i32 %7, 1
$i328B

	full_text


i32 %7
8sext8B.
,
	full_text

%103 = sext i32 %102 to i64
&i328B

	full_text


i32 %102
^getelementptr8BK
I
	full_text<
:
8%104 = getelementptr inbounds float, float* %2, i64 %103
&i648B

	full_text


i64 %103
Bbitcast8B5
3
	full_text&
$
"%105 = bitcast float* %104 to i32*
,float*8B

	full_text

float* %104
Jload8B@
>
	full_text1
/
-%106 = load i32, i32* %105, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %105
7sext8B-
+
	full_text

%107 = sext i32 %17 to i64
%i328B

	full_text
	
i32 %17
^getelementptr8BK
I
	full_text<
:
8%108 = getelementptr inbounds float, float* %0, i64 %107
&i648B

	full_text


i64 %107
Bbitcast8B5
3
	full_text&
$
"%109 = bitcast float* %108 to i32*
,float*8B

	full_text

float* %108
Jstore8B?
=
	full_text0
.
,store i32 %106, i32* %109, align 4, !tbaa !8
&i328B

	full_text


i32 %106
(i32*8B

	full_text

	i32* %109
(br8B 

	full_text

br label %110
<br8B4
2
	full_text%
#
!br i1 %30, label %111, label %121
#i18B

	full_text


i1 %30
4shl8B+
)
	full_text

%112 = shl nsw i32 %7, 1
$i328B

	full_text


i32 %7
0or8B(
&
	full_text

%113 = or i32 %112, 1
&i328B

	full_text


i32 %112
8sext8B.
,
	full_text

%114 = sext i32 %113 to i64
&i328B

	full_text


i32 %113
^getelementptr8BK
I
	full_text<
:
8%115 = getelementptr inbounds float, float* %2, i64 %114
&i648B

	full_text


i64 %114
Bbitcast8B5
3
	full_text&
$
"%116 = bitcast float* %115 to i32*
,float*8B

	full_text

float* %115
Jload8B@
>
	full_text1
/
-%117 = load i32, i32* %116, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %116
7zext8B-
+
	full_text

%118 = zext i32 %29 to i64
%i328B

	full_text
	
i32 %29
^getelementptr8BK
I
	full_text<
:
8%119 = getelementptr inbounds float, float* %0, i64 %118
&i648B

	full_text


i64 %118
Bbitcast8B5
3
	full_text&
$
"%120 = bitcast float* %119 to i32*
,float*8B

	full_text

float* %119
Jstore8B?
=
	full_text0
.
,store i32 %117, i32* %120, align 4, !tbaa !8
&i328B

	full_text


i32 %117
(i32*8B

	full_text

	i32* %120
(br8B 

	full_text

br label %121
$ret8B

	full_text


ret void
*float*8B

	full_text

	float* %1
*float*8B

	full_text

	float* %2
*float*8B

	full_text

	float* %0
$i328B

	full_text


i32 %3
$i328B
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
#i328B

	full_text	

i32 1
$i328B

	full_text


i32 -1
#i328B

	full_text	

i32 0
#i328B

	full_text	

i32 2       	  
 
 

                    !    "# "" $% $$ &' && () (( *+ ** ,- ,. ,, /1 00 23 22 45 47 66 89 88 :; :: <= << >? >> @A @@ BC BB DE DD FG FF HI HJ HH KL MN MM OP OR QQ ST SS UV UU WY XX ZZ [\ [^ ]_ ]] `a `` bb cd ce cc fg fi hj hh kl kk mn mo mm pq pp rs rr tu tt vw vv xy xx z{ zz |} || ~ ~	Ä ~~ ÅÇ Å
É ÅÅ ÑÜ ÖÖ áà áá âä ââ ãå ãé çç èê èè ëí ë
ì ëë îï îî ñó ññ òô òõ öö úû ù
ü ùù †
° †† ¢£ ¢¢ §• §§ ¶¶ ß® ß
© ßß ™´ ™≠ ¨
Æ ¨¨ Ø∞ ØØ ±≤ ±± ≥¥ ≥
µ ≥≥ ∂∑ ∂∂ ∏
π ∏∏ ∫ª ∫∫ ºΩ ºº æ
ø ææ ¿¡ ¿¿ ¬√ ¬
ƒ ¬¬ ≈∆ ≈
« ≈≈ »  …… ÀÃ À
Õ ÀÀ Œœ Œ– —“ —‘ ”” ’÷ ’’ ◊
ÿ ◊◊ Ÿ⁄ ŸŸ €‹ €€ ›ﬁ ›› ﬂ
‡ ﬂﬂ ·‚ ·· „‰ „
Â „„ ÊË ÁÍ ÈÈ ÎÏ ÎÎ ÌÓ ÌÌ Ô
 ÔÔ ÒÚ ÒÒ ÛÙ ÛÛ ıˆ ıı ˜
¯ ˜˜ ˘˙ ˘˘ ˚¸ ˚
˝ ˚˚ ˛Ä Ä 8Å (Å DÅ tÅ zÅ ∏Å æÅ ◊Å ÔÇ ﬂÇ ˜	É 	É 2É LÉ Z
É ç	Ñ 	Ñ 0   	  
           !  # %$ '& )( +" -* . 10 32 50 76 98 ;: = ?> A@ CB ED G< IF JL NM P RQ TQ VÖ YZ \L ^á _Ö a] d ec g` iS jh l` nU om qk sr ut wp yx {z }v | Ä~ Çz É` Ü] àá äâ åX éç êX íè ìë ïî óñ ô õë û§ ü… °† £ù •¢ ® ©ß ´§ ≠ö Æ¨ ∞ù ≤Ø ¥± µØ ∑∂ π∏ ª≥ Ωº øæ ¡∫ √¿ ƒ¬ ∆æ «¢  … Ãî ÕÀ œ “ ‘” ÷’ ÿ◊ ⁄Ÿ ‹ ﬁ› ‡ﬂ ‚€ ‰· Â2 Ë ÍÈ ÏÎ ÓÌ Ô ÚÒ Ù0 ˆı ¯˜ ˙Û ¸˘ ˝  0/ 04 64 LK LO XO Q[ ç[ –W ]ò öò –— ”— Áf hf Öú ùÊ ÁÁ ÈÁ ˇÑ Öã Xã ]™ ¨™ …˛ ˇ» …Œ ùŒ – ÜÜ ˇ áá àà ÖÖ ÖÖ – àà – ÜÜ b àà b áá ¶ àà ¶	â 	â 	â $	â >	â @	â L	â Q	â Sâ Xâ `â b
â Ö
â á
â î
â ö
â ¢
â §â ¶
â …â –
â ”
â È
â Î	ä 	ä k	ä p
ä Øã ã ã 	ã M
ã â
ã ñã †	å U	å Z
å ±"
group_prefixSum"
_Z12get_local_idj"
_Z14get_local_sizej"
_Z12get_group_idj"
_Z7barrierj*ç
PrefixSum_Kernels.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282
 
transfer_bytes_log1p
íA

wgsize
Ä

wgsize_log1p
íA

devmap_label
 

transfer_bytes
Ä 