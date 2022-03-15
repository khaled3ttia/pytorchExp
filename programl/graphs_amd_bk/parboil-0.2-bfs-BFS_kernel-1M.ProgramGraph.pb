

[external]
KcallBC
A
	full_text4
2
0%14 = tail call i64 @_Z12get_local_idj(i32 0) #4
3icmpB+
)
	full_text

%15 = icmp eq i64 %14, 0
#i64B

	full_text
	
i64 %14
8brB2
0
	full_text#
!
br i1 %15, label %16, label %17
!i1B

	full_text


i1 %15
Fstore8B;
9
	full_text,
*
(store i32 0, i32* %10, align 4, !tbaa !8
'br8B

	full_text

br label %17
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
Ncall8BD
B
	full_text5
3
1%18 = tail call i64 @_Z13get_global_idj(i32 0) #4
8trunc8B-
+
	full_text

%19 = trunc i64 %18 to i32
%i648B

	full_text
	
i64 %18
7icmp8B-
+
	full_text

%20 = icmp slt i32 %19, %7
%i328B

	full_text
	
i32 %19
:br8B2
0
	full_text#
!
br i1 %20, label %21, label %64
#i18B

	full_text


i1 %20
1shl8B(
&
	full_text

%22 = shl i64 %18, 32
%i648B

	full_text
	
i64 %18
9ashr8B/
-
	full_text 

%23 = ashr exact i64 %22, 32
%i648B

	full_text
	
i64 %22
Xgetelementptr8BE
C
	full_text6
4
2%24 = getelementptr inbounds i32, i32* %0, i64 %23
%i648B

	full_text
	
i64 %23
Hload8B>
<
	full_text/
-
+%25 = load i32, i32* %24, align 4, !tbaa !8
'i32*8B

	full_text


i32* %24
6sext8B,
*
	full_text

%26 = sext i32 %25 to i64
%i328B

	full_text
	
i32 %25
Xgetelementptr8BE
C
	full_text6
4
2%27 = getelementptr inbounds i32, i32* %4, i64 %26
%i648B

	full_text
	
i64 %26
Mstore8BB
@
	full_text3
1
/store i32 16677221, i32* %27, align 4, !tbaa !8
'i32*8B

	full_text


i32* %27
Xgetelementptr8BE
C
	full_text6
4
2%28 = getelementptr inbounds i32, i32* %5, i64 %26
%i648B

	full_text
	
i64 %26
Hload8B>
<
	full_text/
-
+%29 = load i32, i32* %28, align 4, !tbaa !8
'i32*8B

	full_text


i32* %28
qgetelementptr8B^
\
	full_textO
M
K%30 = getelementptr inbounds %struct.Node, %struct.Node* %2, i64 %26, i32 0
%i648B

	full_text
	
i64 %26
>load8B4
2
	full_text%
#
!%31 = load i32, i32* %30, align 4
'i32*8B

	full_text


i32* %30
qgetelementptr8B^
\
	full_textO
M
K%32 = getelementptr inbounds %struct.Node, %struct.Node* %2, i64 %26, i32 1
%i648B

	full_text
	
i64 %26
>load8B4
2
	full_text%
#
!%33 = load i32, i32* %32, align 4
'i32*8B

	full_text


i32* %32
6icmp8B,
*
	full_text

%34 = icmp sgt i32 %33, 0
%i328B

	full_text
	
i32 %33
:br8B2
0
	full_text#
!
br i1 %34, label %35, label %64
#i18B

	full_text


i1 %34
6add8B-
+
	full_text

%36 = add nsw i32 %33, %31
%i328B

	full_text
	
i32 %33
%i328B

	full_text
	
i32 %31
6sext8B,
*
	full_text

%37 = sext i32 %31 to i64
%i328B

	full_text
	
i32 %31
6sext8B,
*
	full_text

%38 = sext i32 %36 to i64
%i328B

	full_text
	
i32 %36
'br8B

	full_text

br label %39
Dphi8B;
9
	full_text,
*
(%40 = phi i64 [ %37, %35 ], [ %62, %61 ]
%i648B

	full_text
	
i64 %37
%i648B

	full_text
	
i64 %62
qgetelementptr8B^
\
	full_textO
M
K%41 = getelementptr inbounds %struct.Edge, %struct.Edge* %3, i64 %40, i32 0
%i648B

	full_text
	
i64 %40
>load8B4
2
	full_text%
#
!%42 = load i32, i32* %41, align 4
'i32*8B

	full_text


i32* %41
qgetelementptr8B^
\
	full_textO
M
K%43 = getelementptr inbounds %struct.Edge, %struct.Edge* %3, i64 %40, i32 1
%i648B

	full_text
	
i64 %40
>load8B4
2
	full_text%
#
!%44 = load i32, i32* %43, align 4
'i32*8B

	full_text


i32* %43
6add8B-
+
	full_text

%45 = add nsw i32 %44, %29
%i328B

	full_text
	
i32 %44
%i328B

	full_text
	
i32 %29
6sext8B,
*
	full_text

%46 = sext i32 %42 to i64
%i328B

	full_text
	
i32 %42
Xgetelementptr8BE
C
	full_text6
4
2%47 = getelementptr inbounds i32, i32* %5, i64 %46
%i648B

	full_text
	
i64 %46
acall8BW
U
	full_textH
F
D%48 = tail call i32 @_Z8atom_minPU8CLglobalVii(i32* %47, i32 %45) #6
'i32*8B

	full_text


i32* %47
%i328B

	full_text
	
i32 %45
8icmp8B.
,
	full_text

%49 = icmp sgt i32 %48, %45
%i328B

	full_text
	
i32 %48
%i328B

	full_text
	
i32 %45
:br8B2
0
	full_text#
!
br i1 %49, label %50, label %61
#i18B

	full_text


i1 %49
Xgetelementptr8BE
C
	full_text6
4
2%51 = getelementptr inbounds i32, i32* %4, i64 %46
%i648B

	full_text
	
i64 %46
Hload8B>
<
	full_text/
-
+%52 = load i32, i32* %51, align 4, !tbaa !8
'i32*8B

	full_text


i32* %51
=icmp8B3
1
	full_text$
"
 %53 = icmp sgt i32 %52, 16677216
%i328B

	full_text
	
i32 %52
:br8B2
0
	full_text#
!
br i1 %53, label %54, label %61
#i18B

	full_text


i1 %53
acall8BW
U
	full_textH
F
D%55 = tail call i32 @_Z9atom_xchgPU8CLglobalVii(i32* %51, i32 %8) #6
'i32*8B

	full_text


i32* %51
6icmp8B,
*
	full_text

%56 = icmp eq i32 %55, %8
%i328B

	full_text
	
i32 %55
:br8B2
0
	full_text#
!
br i1 %56, label %61, label %57
#i18B

	full_text


i1 %56
^call8BT
R
	full_textE
C
A%58 = tail call i32 @_Z8atom_addPU7CLlocalVii(i32* %10, i32 1) #6
6sext8B,
*
	full_text

%59 = sext i32 %58 to i64
%i328B

	full_text
	
i32 %58
Ygetelementptr8BF
D
	full_text7
5
3%60 = getelementptr inbounds i32, i32* %11, i64 %59
%i648B

	full_text
	
i64 %59
Hstore8B=
;
	full_text.
,
*store i32 %42, i32* %60, align 4, !tbaa !8
%i328B

	full_text
	
i32 %42
'i32*8B

	full_text


i32* %60
'br8B

	full_text

br label %61
4add8	B+
)
	full_text

%62 = add nsw i64 %40, 1
%i648	B

	full_text
	
i64 %40
8icmp8	B.
,
	full_text

%63 = icmp slt i64 %62, %38
%i648	B

	full_text
	
i64 %62
%i648	B

	full_text
	
i64 %38
:br8	B2
0
	full_text#
!
br i1 %63, label %39, label %64
#i18	B

	full_text


i1 %63
Bcall8
B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
:br8
B2
0
	full_text#
!
br i1 %15, label %65, label %68
#i18
B

	full_text


i1 %15
Hload8B>
<
	full_text/
-
+%66 = load i32, i32* %10, align 4, !tbaa !8
`call8BV
T
	full_textG
E
C%67 = tail call i32 @_Z8atom_addPU8CLglobalVii(i32* %6, i32 %66) #6
%i328B

	full_text
	
i32 %66
Hstore8B=
;
	full_text.
,
*store i32 %67, i32* %12, align 4, !tbaa !8
%i328B

	full_text
	
i32 %67
'br8B

	full_text

br label %68
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
8trunc8B-
+
	full_text

%69 = trunc i64 %14 to i32
%i648B

	full_text
	
i64 %14
Hload8B>
<
	full_text/
-
+%70 = load i32, i32* %10, align 4, !tbaa !8
8icmp8B.
,
	full_text

%71 = icmp sgt i32 %70, %69
%i328B

	full_text
	
i32 %70
%i328B

	full_text
	
i32 %69
:br8B2
0
	full_text#
!
br i1 %71, label %72, label %89
#i18B

	full_text


i1 %71
Ocall8BE
C
	full_text6
4
2%73 = tail call i64 @_Z14get_local_sizej(i32 0) #4
'br8B

	full_text

br label %74
Dphi8B;
9
	full_text,
*
(%75 = phi i32 [ %69, %72 ], [ %86, %74 ]
%i328B

	full_text
	
i32 %69
%i328B

	full_text
	
i32 %86
Dphi8B;
9
	full_text,
*
(%76 = phi i64 [ %14, %72 ], [ %85, %74 ]
%i648B

	full_text
	
i64 %14
%i648B

	full_text
	
i64 %85
1shl8B(
&
	full_text

%77 = shl i64 %76, 32
%i648B

	full_text
	
i64 %76
9ashr8B/
-
	full_text 

%78 = ashr exact i64 %77, 32
%i648B

	full_text
	
i64 %77
Ygetelementptr8BF
D
	full_text7
5
3%79 = getelementptr inbounds i32, i32* %11, i64 %78
%i648B

	full_text
	
i64 %78
Hload8B>
<
	full_text/
-
+%80 = load i32, i32* %79, align 4, !tbaa !8
'i32*8B

	full_text


i32* %79
Hload8B>
<
	full_text/
-
+%81 = load i32, i32* %12, align 4, !tbaa !8
6add8B-
+
	full_text

%82 = add nsw i32 %81, %75
%i328B

	full_text
	
i32 %81
%i328B

	full_text
	
i32 %75
6sext8B,
*
	full_text

%83 = sext i32 %82 to i64
%i328B

	full_text
	
i32 %82
Xgetelementptr8BE
C
	full_text6
4
2%84 = getelementptr inbounds i32, i32* %1, i64 %83
%i648B

	full_text
	
i64 %83
Hstore8B=
;
	full_text.
,
*store i32 %80, i32* %84, align 4, !tbaa !8
%i328B

	full_text
	
i32 %80
'i32*8B

	full_text


i32* %84
2add8B)
'
	full_text

%85 = add i64 %73, %78
%i648B

	full_text
	
i64 %73
%i648B

	full_text
	
i64 %78
8trunc8B-
+
	full_text

%86 = trunc i64 %85 to i32
%i648B

	full_text
	
i64 %85
Hload8B>
<
	full_text/
-
+%87 = load i32, i32* %10, align 4, !tbaa !8
8icmp8B.
,
	full_text

%88 = icmp sgt i32 %87, %86
%i328B

	full_text
	
i32 %87
%i328B

	full_text
	
i32 %86
:br8B2
0
	full_text#
!
br i1 %88, label %74, label %89
#i18B

	full_text


i1 %88
$ret8B

	full_text


ret void
2struct*8B#
!
	full_text

%struct.Edge* %3
$i328B

	full_text


i32 %8
'i32*8B

	full_text


i32* %10
'i32*8B

	full_text


i32* %12
&i32*8B

	full_text
	
i32* %4
'i32*8B

	full_text


i32* %11
$i328B

	full_text


i32 %7
&i32*8B

	full_text
	
i32* %5
2struct*8B#
!
	full_text

%struct.Node* %2
&i32*8B

	full_text
	
i32* %6
&i32*8B

	full_text
	
i32* %0
&i32*8B

	full_text
	
i32* %1
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
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 0
#i328B

	full_text	

i32 1
$i648B

	full_text


i64 32
*i328B

	full_text

i32 16677221
*i328B

	full_text

i32 16677216
#i648B

	full_text	

i64 1      		 
 

                     !    "# "" $% $$ &' && () (( *+ ** ,- ,/ .0 .. 12 11 34 33 57 68 66 9: 99 ;< ;; => == ?@ ?? AB AC AA DE DD FG FF HI HJ HH KL KM KK NO NQ PP RS RR TU TT VW VY XX Z[ ZZ \] \^ _` __ ab aa cd ce cc fh gg ij ik ii lm ln op oq rs rr tu tt vw xy xx zz {| {} {{ ~ ~Ä ÅÉ Ç
Ñ ÇÇ ÖÜ Ö
á ÖÖ àâ àà äã ää å
ç åå éè éé êê ëí ë
ì ëë îï îî ñ
ó ññ òô ò
ö òò õú õ
ù õõ ûü ûû †† °¢ °
£ °° §• §ß 9ß =	® X	® Z	© © ^© q© z© †	™ t™ ê´ ´ P¨ a¨ å	≠ Æ Æ FØ "Ø &∞ r± ≤ ñ  	 
  	         ! #" % '& )( +* -( /$ 0$ 2. 41 7g 86 :9 <6 >= @? B  C; ED GF IA JH LA MK OD QP SR UT WP YX [Z ]^ `_ b; da e6 hg j3 ki m pq sr u yz |x }{ x Éû Ñ Üõ áÖ âà ãä çå èê íÇ ìë ïî óé ôñ öÄ úä ùõ ü† ¢û £° •     n, ., no qo w5 6v w~ Ä~ ¶N PN gÅ ÇV XV gl 6l n§ Ç§ ¶\ g\ ^f g µµ ¶ ∂∂ ∑∑ ≥≥ ∏∏ ¥¥ ∫∫ ππ	 µµ 	 ¥¥ H ∂∂ Hr ππ rÄ ∫∫ Ä^ ∏∏ ^X ∑∑ Xw ¥¥ w ≥≥ n ¥¥ nª ª ª 		ª "	ª *	ª 9ª Ä	º Ω 	Ω &	Ω =	Ω ^Ω nΩ w	æ 	æ 
æ à
æ äø 	¿ T	¡ g"

BFS_kernel"
_Z12get_local_idj"
_Z7barrierj"
_Z13get_global_idj"
_Z8atom_minPU8CLglobalVii"
_Z9atom_xchgPU8CLglobalVii"
_Z8atom_addPU7CLlocalVii"
_Z8atom_addPU8CLglobalVii"
_Z14get_local_sizej*Ü
BFS_kernel.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Å

wgsize_log1p
ı∫êA

transfer_bytes
î–¶"
 
transfer_bytes_log1p
ı∫êA

devmap_label
 

wgsize
ﬂ