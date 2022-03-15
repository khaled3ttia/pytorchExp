
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
-shlB&
$
	full_text

%9 = shl i64 %7, 32
"i64B

	full_text


i64 %7
6ashrB.
,
	full_text

%10 = ashr exact i64 %9, 32
"i64B

	full_text


i64 %9
VgetelementptrBE
C
	full_text6
4
2%11 = getelementptr inbounds i32, i32* %2, i64 %10
#i64B

	full_text
	
i64 %10
FloadB>
<
	full_text/
-
+%12 = load i32, i32* %11, align 4, !tbaa !8
%i32*B

	full_text


i32* %11
5icmpB-
+
	full_text

%13 = icmp slt i32 %12, %4
#i32B

	full_text
	
i32 %12
9brB3
1
	full_text$
"
 br i1 %13, label %14, label %131
!i1B

	full_text


i1 %13
Pcall8BF
D
	full_text7
5
3%15 = tail call i64 @_Z15get_global_sizej(i32 0) #2
8trunc8B-
+
	full_text

%16 = trunc i64 %15 to i32
%i648B

	full_text
	
i64 %15
Xgetelementptr8BE
C
	full_text6
4
2%17 = getelementptr inbounds i32, i32* %3, i64 %10
%i648B

	full_text
	
i64 %10
Hload8B>
<
	full_text/
-
+%18 = load i32, i32* %17, align 4, !tbaa !8
'i32*8B

	full_text


i32* %17
7icmp8B-
+
	full_text

%19 = icmp slt i32 %8, %16
$i328B

	full_text


i32 %8
%i328B

	full_text
	
i32 %16
5icmp8B+
)
	full_text

%20 = icmp sgt i32 %8, 0
$i328B

	full_text


i32 %8
1and8B(
&
	full_text

%21 = and i1 %19, %20
#i18B

	full_text


i1 %19
#i18B

	full_text


i1 %20
;br8B3
1
	full_text$
"
 br i1 %21, label %22, label %131
#i18B

	full_text


i1 %21
8and8B/
-
	full_text 

%23 = and i64 %7, 4294967295
$i648B

	full_text


i64 %7
5add8B,
*
	full_text

%24 = add nsw i64 %23, -1
%i648B

	full_text
	
i64 %23
/and8B&
$
	full_text

%25 = and i64 %7, 7
$i648B

	full_text


i64 %7
6icmp8B,
*
	full_text

%26 = icmp ult i64 %24, 7
%i648B

	full_text
	
i64 %24
:br8B2
0
	full_text#
!
br i1 %26, label %67, label %27
#i18B

	full_text


i1 %26
6sub8B-
+
	full_text

%28 = sub nsw i64 %23, %25
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %25
'br8B

	full_text

br label %29
Bphi8B9
7
	full_text*
(
&%30 = phi i64 [ 0, %27 ], [ %64, %29 ]
%i648B

	full_text
	
i64 %64
Bphi8B9
7
	full_text*
(
&%31 = phi i32 [ 0, %27 ], [ %63, %29 ]
%i328B

	full_text
	
i32 %63
Dphi8B;
9
	full_text,
*
(%32 = phi i64 [ %28, %27 ], [ %65, %29 ]
%i648B

	full_text
	
i64 %28
%i648B

	full_text
	
i64 %65
Xgetelementptr8BE
C
	full_text6
4
2%33 = getelementptr inbounds i32, i32* %1, i64 %30
%i648B

	full_text
	
i64 %30
Hload8B>
<
	full_text/
-
+%34 = load i32, i32* %33, align 4, !tbaa !8
'i32*8B

	full_text


i32* %33
6add8B-
+
	full_text

%35 = add nsw i32 %34, %31
%i328B

	full_text
	
i32 %34
%i328B

	full_text
	
i32 %31
.or8B&
$
	full_text

%36 = or i64 %30, 1
%i648B

	full_text
	
i64 %30
Xgetelementptr8BE
C
	full_text6
4
2%37 = getelementptr inbounds i32, i32* %1, i64 %36
%i648B

	full_text
	
i64 %36
Hload8B>
<
	full_text/
-
+%38 = load i32, i32* %37, align 4, !tbaa !8
'i32*8B

	full_text


i32* %37
6add8B-
+
	full_text

%39 = add nsw i32 %38, %35
%i328B

	full_text
	
i32 %38
%i328B

	full_text
	
i32 %35
.or8B&
$
	full_text

%40 = or i64 %30, 2
%i648B

	full_text
	
i64 %30
Xgetelementptr8BE
C
	full_text6
4
2%41 = getelementptr inbounds i32, i32* %1, i64 %40
%i648B

	full_text
	
i64 %40
Hload8B>
<
	full_text/
-
+%42 = load i32, i32* %41, align 4, !tbaa !8
'i32*8B

	full_text


i32* %41
6add8B-
+
	full_text

%43 = add nsw i32 %42, %39
%i328B

	full_text
	
i32 %42
%i328B

	full_text
	
i32 %39
.or8B&
$
	full_text

%44 = or i64 %30, 3
%i648B

	full_text
	
i64 %30
Xgetelementptr8BE
C
	full_text6
4
2%45 = getelementptr inbounds i32, i32* %1, i64 %44
%i648B

	full_text
	
i64 %44
Hload8B>
<
	full_text/
-
+%46 = load i32, i32* %45, align 4, !tbaa !8
'i32*8B

	full_text


i32* %45
6add8B-
+
	full_text

%47 = add nsw i32 %46, %43
%i328B

	full_text
	
i32 %46
%i328B

	full_text
	
i32 %43
.or8B&
$
	full_text

%48 = or i64 %30, 4
%i648B

	full_text
	
i64 %30
Xgetelementptr8BE
C
	full_text6
4
2%49 = getelementptr inbounds i32, i32* %1, i64 %48
%i648B

	full_text
	
i64 %48
Hload8B>
<
	full_text/
-
+%50 = load i32, i32* %49, align 4, !tbaa !8
'i32*8B

	full_text


i32* %49
6add8B-
+
	full_text

%51 = add nsw i32 %50, %47
%i328B

	full_text
	
i32 %50
%i328B

	full_text
	
i32 %47
.or8B&
$
	full_text

%52 = or i64 %30, 5
%i648B

	full_text
	
i64 %30
Xgetelementptr8BE
C
	full_text6
4
2%53 = getelementptr inbounds i32, i32* %1, i64 %52
%i648B

	full_text
	
i64 %52
Hload8B>
<
	full_text/
-
+%54 = load i32, i32* %53, align 4, !tbaa !8
'i32*8B

	full_text


i32* %53
6add8B-
+
	full_text

%55 = add nsw i32 %54, %51
%i328B

	full_text
	
i32 %54
%i328B

	full_text
	
i32 %51
.or8B&
$
	full_text

%56 = or i64 %30, 6
%i648B

	full_text
	
i64 %30
Xgetelementptr8BE
C
	full_text6
4
2%57 = getelementptr inbounds i32, i32* %1, i64 %56
%i648B

	full_text
	
i64 %56
Hload8B>
<
	full_text/
-
+%58 = load i32, i32* %57, align 4, !tbaa !8
'i32*8B

	full_text


i32* %57
6add8B-
+
	full_text

%59 = add nsw i32 %58, %55
%i328B

	full_text
	
i32 %58
%i328B

	full_text
	
i32 %55
.or8B&
$
	full_text

%60 = or i64 %30, 7
%i648B

	full_text
	
i64 %30
Xgetelementptr8BE
C
	full_text6
4
2%61 = getelementptr inbounds i32, i32* %1, i64 %60
%i648B

	full_text
	
i64 %60
Hload8B>
<
	full_text/
-
+%62 = load i32, i32* %61, align 4, !tbaa !8
'i32*8B

	full_text


i32* %61
6add8B-
+
	full_text

%63 = add nsw i32 %62, %59
%i328B

	full_text
	
i32 %62
%i328B

	full_text
	
i32 %59
4add8B+
)
	full_text

%64 = add nsw i64 %30, 8
%i648B

	full_text
	
i64 %30
1add8B(
&
	full_text

%65 = add i64 %32, -8
%i648B

	full_text
	
i64 %32
5icmp8B+
)
	full_text

%66 = icmp eq i64 %65, 0
%i648B

	full_text
	
i64 %65
:br8B2
0
	full_text#
!
br i1 %66, label %67, label %29
#i18B

	full_text


i1 %66
Fphi8B=
;
	full_text.
,
*%68 = phi i32 [ undef, %22 ], [ %63, %29 ]
%i328B

	full_text
	
i32 %63
Bphi8B9
7
	full_text*
(
&%69 = phi i64 [ 0, %22 ], [ %64, %29 ]
%i648B

	full_text
	
i64 %64
Bphi8B9
7
	full_text*
(
&%70 = phi i32 [ 0, %22 ], [ %63, %29 ]
%i328B

	full_text
	
i32 %63
5icmp8B+
)
	full_text

%71 = icmp eq i64 %25, 0
%i648B

	full_text
	
i64 %25
:br8B2
0
	full_text#
!
br i1 %71, label %83, label %72
#i18B

	full_text


i1 %71
'br8B

	full_text

br label %73
Dphi8B;
9
	full_text,
*
(%74 = phi i64 [ %69, %72 ], [ %80, %73 ]
%i648B

	full_text
	
i64 %69
%i648B

	full_text
	
i64 %80
Dphi8B;
9
	full_text,
*
(%75 = phi i32 [ %70, %72 ], [ %79, %73 ]
%i328B

	full_text
	
i32 %70
%i328B

	full_text
	
i32 %79
Dphi8B;
9
	full_text,
*
(%76 = phi i64 [ %25, %72 ], [ %81, %73 ]
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %81
Xgetelementptr8BE
C
	full_text6
4
2%77 = getelementptr inbounds i32, i32* %1, i64 %74
%i648B

	full_text
	
i64 %74
Hload8B>
<
	full_text/
-
+%78 = load i32, i32* %77, align 4, !tbaa !8
'i32*8B

	full_text


i32* %77
6add8B-
+
	full_text

%79 = add nsw i32 %78, %75
%i328B

	full_text
	
i32 %78
%i328B

	full_text
	
i32 %75
8add8B/
-
	full_text 

%80 = add nuw nsw i64 %74, 1
%i648B

	full_text
	
i64 %74
1add8B(
&
	full_text

%81 = add i64 %76, -1
%i648B

	full_text
	
i64 %76
5icmp8B+
)
	full_text

%82 = icmp eq i64 %81, 0
%i648B

	full_text
	
i64 %81
Jbr8BB
@
	full_text3
1
/br i1 %82, label %83, label %73, !llvm.loop !12
#i18B

	full_text


i1 %82
Dphi8B;
9
	full_text,
*
(%84 = phi i32 [ %68, %67 ], [ %79, %73 ]
%i328B

	full_text
	
i32 %68
%i328B

	full_text
	
i32 %79
6icmp8B,
*
	full_text

%85 = icmp sgt i32 %84, 0
%i328B

	full_text
	
i32 %84
;br8B3
1
	full_text$
"
 br i1 %85, label %86, label %131
#i18B

	full_text


i1 %85
5sext8	B+
)
	full_text

%87 = sext i32 %5 to i64
Xgetelementptr8	BE
C
	full_text6
4
2%88 = getelementptr inbounds i32, i32* %0, i64 %87
%i648	B

	full_text
	
i64 %87
8icmp8	B.
,
	full_text

%89 = icmp slt i32 %12, %18
%i328	B

	full_text
	
i32 %12
%i328	B

	full_text
	
i32 %18
;br8	B3
1
	full_text$
"
 br i1 %89, label %90, label %131
#i18	B

	full_text


i1 %89
6sext8
B,
*
	full_text

%91 = sext i32 %12 to i64
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
%92 = sext i32 %18 to i64
%i328
B

	full_text
	
i32 %18
6sub8
B-
+
	full_text

%93 = sub nsw i64 %92, %91
%i648
B

	full_text
	
i64 %92
%i648
B

	full_text
	
i64 %91
5add8
B,
*
	full_text

%94 = add nsw i64 %92, -1
%i648
B

	full_text
	
i64 %92
6sub8
B-
+
	full_text

%95 = sub nsw i64 %94, %91
%i648
B

	full_text
	
i64 %94
%i648
B

	full_text
	
i64 %91
0and8
B'
%
	full_text

%96 = and i64 %93, 3
%i648
B

	full_text
	
i64 %93
5icmp8
B+
)
	full_text

%97 = icmp eq i64 %96, 0
%i648
B

	full_text
	
i64 %96
;br8
B3
1
	full_text$
"
 br i1 %97, label %108, label %98
#i18
B

	full_text


i1 %97
'br8B

	full_text

br label %99
Fphi8B=
;
	full_text.
,
*%100 = phi i64 [ %91, %98 ], [ %105, %99 ]
%i648B

	full_text
	
i64 %91
&i648B

	full_text


i64 %105
Fphi8B=
;
	full_text.
,
*%101 = phi i64 [ %96, %98 ], [ %106, %99 ]
%i648B

	full_text
	
i64 %96
&i648B

	full_text


i64 %106
[getelementptr8BH
F
	full_text9
7
5%102 = getelementptr inbounds i32, i32* %88, i64 %100
'i32*8B

	full_text


i32* %88
&i648B

	full_text


i64 %100
Jload8B@
>
	full_text1
/
-%103 = load i32, i32* %102, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %102
8add8B/
-
	full_text 

%104 = add nsw i32 %103, %84
&i328B

	full_text


i32 %103
%i328B

	full_text
	
i32 %84
Jstore8B?
=
	full_text0
.
,store i32 %104, i32* %102, align 4, !tbaa !8
&i328B

	full_text


i32 %104
(i32*8B

	full_text

	i32* %102
6add8B-
+
	full_text

%105 = add nsw i64 %100, 1
&i648B

	full_text


i64 %100
3add8B*
(
	full_text

%106 = add i64 %101, -1
&i648B

	full_text


i64 %101
7icmp8B-
+
	full_text

%107 = icmp eq i64 %106, 0
&i648B

	full_text


i64 %106
Lbr8BD
B
	full_text5
3
1br i1 %107, label %108, label %99, !llvm.loop !14
$i18B

	full_text
	
i1 %107
Fphi8B=
;
	full_text.
,
*%109 = phi i64 [ %91, %90 ], [ %105, %99 ]
%i648B

	full_text
	
i64 %91
&i648B

	full_text


i64 %105
7icmp8B-
+
	full_text

%110 = icmp ult i64 %95, 3
%i648B

	full_text
	
i64 %95
=br8B5
3
	full_text&
$
"br i1 %110, label %131, label %111
$i18B

	full_text
	
i1 %110
(br8B 

	full_text

br label %112
Iphi8B@
>
	full_text1
/
-%113 = phi i64 [ %109, %111 ], [ %129, %112 ]
&i648B

	full_text


i64 %109
&i648B

	full_text


i64 %129
[getelementptr8BH
F
	full_text9
7
5%114 = getelementptr inbounds i32, i32* %88, i64 %113
'i32*8B

	full_text


i32* %88
&i648B

	full_text


i64 %113
Jload8B@
>
	full_text1
/
-%115 = load i32, i32* %114, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %114
8add8B/
-
	full_text 

%116 = add nsw i32 %115, %84
&i328B

	full_text


i32 %115
%i328B

	full_text
	
i32 %84
Jstore8B?
=
	full_text0
.
,store i32 %116, i32* %114, align 4, !tbaa !8
&i328B

	full_text


i32 %116
(i32*8B

	full_text

	i32* %114
6add8B-
+
	full_text

%117 = add nsw i64 %113, 1
&i648B

	full_text


i64 %113
[getelementptr8BH
F
	full_text9
7
5%118 = getelementptr inbounds i32, i32* %88, i64 %117
'i32*8B

	full_text


i32* %88
&i648B

	full_text


i64 %117
Jload8B@
>
	full_text1
/
-%119 = load i32, i32* %118, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %118
8add8B/
-
	full_text 

%120 = add nsw i32 %119, %84
&i328B

	full_text


i32 %119
%i328B

	full_text
	
i32 %84
Jstore8B?
=
	full_text0
.
,store i32 %120, i32* %118, align 4, !tbaa !8
&i328B

	full_text


i32 %120
(i32*8B

	full_text

	i32* %118
6add8B-
+
	full_text

%121 = add nsw i64 %113, 2
&i648B

	full_text


i64 %113
[getelementptr8BH
F
	full_text9
7
5%122 = getelementptr inbounds i32, i32* %88, i64 %121
'i32*8B

	full_text


i32* %88
&i648B

	full_text


i64 %121
Jload8B@
>
	full_text1
/
-%123 = load i32, i32* %122, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %122
8add8B/
-
	full_text 

%124 = add nsw i32 %123, %84
&i328B

	full_text


i32 %123
%i328B

	full_text
	
i32 %84
Jstore8B?
=
	full_text0
.
,store i32 %124, i32* %122, align 4, !tbaa !8
&i328B

	full_text


i32 %124
(i32*8B

	full_text

	i32* %122
6add8B-
+
	full_text

%125 = add nsw i64 %113, 3
&i648B

	full_text


i64 %113
[getelementptr8BH
F
	full_text9
7
5%126 = getelementptr inbounds i32, i32* %88, i64 %125
'i32*8B

	full_text


i32* %88
&i648B

	full_text


i64 %125
Jload8B@
>
	full_text1
/
-%127 = load i32, i32* %126, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %126
8add8B/
-
	full_text 

%128 = add nsw i32 %127, %84
&i328B

	full_text


i32 %127
%i328B

	full_text
	
i32 %84
Jstore8B?
=
	full_text0
.
,store i32 %128, i32* %126, align 4, !tbaa !8
&i328B

	full_text


i32 %128
(i32*8B

	full_text

	i32* %126
6add8B-
+
	full_text

%129 = add nsw i64 %113, 4
&i648B

	full_text


i64 %113
9icmp8B/
-
	full_text 

%130 = icmp eq i64 %129, %92
&i648B

	full_text


i64 %129
%i648B

	full_text
	
i64 %92
=br8B5
3
	full_text&
$
"br i1 %130, label %131, label %112
$i18B

	full_text
	
i1 %130
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %4
&i32*8B

	full_text
	
i32* %1
&i32*8B

	full_text
	
i32* %0
&i32*8B

	full_text
	
i32* %2
&i32*8B

	full_text
	
i32* %3
$i328B

	full_text


i32 %5
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
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 5
$i648B

	full_text


i64 -8
#i648B

	full_text	

i64 6
$i648B

	full_text


i64 32
$i648B

	full_text


i64 -1
,i648B!

	full_text

i64 4294967295
#i648B

	full_text	

i64 2
#i648B

	full_text	

i64 3
'i328B

	full_text

	i32 undef
#i648B

	full_text	

i64 8
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 4
#i648B

	full_text	

i64 7
#i648B

	full_text	

i64 1       	  
 

                     " !! #$ ## %& %% '( '' )* ), +- ++ .0 // 12 11 34 35 33 67 66 89 88 :; :< :: => == ?@ ?? AB AA CD CE CC FG FF HI HH JK JJ LM LN LL OP OO QR QQ ST SS UV UW UU XY XX Z[ ZZ \] \\ ^_ ^` ^^ ab aa cd cc ef ee gh gi gg jk jj lm ll no nn pq pr pp st ss uv uu wx ww yz y{ yy |} || ~ ~~ € €€ ‚ƒ ‚
… „„ †
‡ †† 
‰  ‹    
‘  ’“ ’
” ’’ •– •
— •• 
™  ›   
     ΅Ά ΅΅ £¤ ££ ¥¦ ¥¨ §
© §§ ª« ªª ¬­ ¬® ―
° ―― ±² ±
³ ±± ΄µ ΄· ¶¶ ΈΉ ΈΈ Ί» Ί
Ό ΊΊ ½Ύ ½½ Ώΐ Ώ
Α ΏΏ ΒΓ ΒΒ ΔΕ ΔΔ ΖΗ ΖΚ Ι
Λ ΙΙ ΜΝ Μ
Ξ ΜΜ ΟΠ Ο
Ρ ΟΟ ÒΣ ÒÒ ΤΥ Τ
Φ ΤΤ ΧΨ Χ
Ω ΧΧ ΪΫ ΪΪ άέ άά ήί ήή ΰα ΰγ β
δ ββ εζ εε ηθ ηλ κ
μ κκ νξ ν
ο νν πρ ππ ςσ ς
τ ςς υφ υ
χ υυ ψω ψψ ϊϋ ϊ
ό ϊϊ ύώ ύύ ÿ€ ÿ
 ÿÿ ‚ƒ ‚
„ ‚‚ …† …… ‡ ‡
‰ ‡‡ ‹   
   
‘  ’“ ’’ ”• ”
– ”” — —— ™ ™
› ™™  
     ΅Ά ΅
£ ΅΅ ¤¥ ¤	§ ¨ 6¨ ?¨ H¨ Q¨ Z¨ c¨ l¨ u¨ © ―ª « ¬ ®    	 
             "! $ &# (' *! ,% -| 0y 2+ 4~ 5/ 76 98 ;1 </ >= @? BA D: E/ GF IH KJ MC N/ PO RQ TS VL W/ YX [Z ]\ _U `/ ba dc fe h^ i/ kj ml on qg r/ ts vu xw zp {/ }3 ~ € ƒy …| ‡y ‰% ‹ †  ‘ “ ”% –΅ — ™ › ’   • Ά΅ ¤£ ¦„ ¨ ©§ «ª ­® °
 ² ³± µ
 · ΉΈ »¶ ΌΈ Ύ½ ΐ¶ ΑΊ ΓΒ ΕΔ Η¶ ΚΪ ΛΒ Νά Ξ― ΠΙ ΡΟ ΣÒ Υ§ ΦΤ ΨΟ ΩΙ ΫΜ έά ίή α¶ γΪ δΏ ζε θβ λ μ― ξκ ον ρπ σ§ τς φν χκ ω― ϋψ όϊ ώύ €§ ÿ ƒϊ „κ †― … ‰‡ ‹ §  ‡ ‘κ “― •’ –” — § ›™ ” κ   ΆΈ £΅ ¥  ¦ ! ¦) „) + § . /¬ ®¬ ¦ ‚ „‚ /΄ ¶΄ ¦¥ §¥ Ζ βΖ Θη ¦η ιΘ Ιι κΰ βΰ Ι¤ ¦¤ κ ¦ ®® ­­ ­­  ®® ― /
― €― †
― 
― £
― Δ
― ή	° a	± ~	² j	³ 	³ 	΄ #
΄ ΅
΄ ½
΄ ά	µ !	¶ F
¶ …	· O
· Β
· ε
· ’Έ „	Ή |Ί Ί 	Ί Ί 1Ί 
Ί ª	» X
» 	Ό %	Ό '	Ό s	½ =
½ 
½ Ϊ
½ ψ"	
makea_5"
_Z13get_global_idj"
_Z15get_global_sizej*
npb-CG-makea_5.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282€

transfer_bytes
όδƒ

wgsize_log1p
½„A

wgsize
 

devmap_label
 
 
transfer_bytes_log1p
½„A