

[external]
KcallBC
A
	full_text4
2
0%9 = tail call i64 @_Z13get_global_idj(i32 0) #3
3zextB+
)
	full_text

%10 = zext i32 %6 to i64
/addB(
&
	full_text

%11 = add i64 %9, %10
"i64B

	full_text


i64 %9
#i64B

	full_text
	
i64 %10
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
0%13 = tail call i64 @_Z12get_local_idj(i32 0) #3
6truncB-
+
	full_text

%14 = trunc i64 %13 to i32
#i64B

	full_text
	
i64 %13
5icmpB-
+
	full_text

%15 = icmp slt i32 %12, %7
#i32B

	full_text
	
i32 %12
8brB2
0
	full_text#
!
br i1 %15, label %16, label %60
!i1B

	full_text


i1 %15
Jbitcast8B=
;
	full_text.
,
*%17 = bitcast double* %0 to [66 x double]*
Jbitcast8B=
;
	full_text.
,
*%18 = bitcast double* %1 to [66 x double]*
6icmp8B,
*
	full_text

%19 = icmp slt i32 %4, %5
:br8B2
0
	full_text#
!
br i1 %19, label %20, label %60
#i18B

	full_text


i1 %19
1shl8B(
&
	full_text

%21 = shl i64 %11, 32
%i648B

	full_text
	
i64 %11
9ashr8B/
-
	full_text 

%22 = ashr exact i64 %21, 32
%i648B

	full_text
	
i64 %21
9add8B0
.
	full_text!

%23 = add i64 %21, 4294967296
%i648B

	full_text
	
i64 %21
9ashr8B/
-
	full_text 

%24 = ashr exact i64 %23, 32
%i648B

	full_text
	
i64 %23
5sext8B+
)
	full_text

%25 = sext i32 %4 to i64
vgetelementptr8Bc
a
	full_textT
R
P%26 = getelementptr inbounds [66 x double], [66 x double]* %17, i64 %22, i64 %25
;[66 x double]*8B%
#
	full_text

[66 x double]* %17
%i648B

	full_text
	
i64 %22
%i648B

	full_text
	
i64 %25
Nload8BD
B
	full_text5
3
1%27 = load double, double* %26, align 8, !tbaa !8
-double*8B

	full_text

double* %26
vgetelementptr8Bc
a
	full_textT
R
P%28 = getelementptr inbounds [66 x double], [66 x double]* %17, i64 %24, i64 %25
;[66 x double]*8B%
#
	full_text

[66 x double]* %17
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %25
Nload8BD
B
	full_text5
3
1%29 = load double, double* %28, align 8, !tbaa !8
-double*8B

	full_text

double* %28
vgetelementptr8Bc
a
	full_textT
R
P%30 = getelementptr inbounds [66 x double], [66 x double]* %18, i64 %22, i64 %25
;[66 x double]*8B%
#
	full_text

[66 x double]* %18
%i648B

	full_text
	
i64 %22
%i648B

	full_text
	
i64 %25
Nload8BD
B
	full_text5
3
1%31 = load double, double* %30, align 8, !tbaa !8
-double*8B

	full_text

double* %30
vgetelementptr8Bc
a
	full_textT
R
P%32 = getelementptr inbounds [66 x double], [66 x double]* %18, i64 %24, i64 %25
;[66 x double]*8B%
#
	full_text

[66 x double]* %18
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %25
Nload8BD
B
	full_text5
3
1%33 = load double, double* %32, align 8, !tbaa !8
-double*8B

	full_text

double* %32
5sext8B+
)
	full_text

%34 = sext i32 %5 to i64
'br8B

	full_text

br label %35
Gphi8B>
<
	full_text/
-
+%36 = phi double [ %33, %20 ], [ %56, %35 ]
+double8B

	full_text


double %33
+double8B

	full_text


double %56
Gphi8B>
<
	full_text/
-
+%37 = phi double [ %31, %20 ], [ %52, %35 ]
+double8B

	full_text


double %31
+double8B

	full_text


double %52
Gphi8B>
<
	full_text/
-
+%38 = phi double [ %29, %20 ], [ %48, %35 ]
+double8B

	full_text


double %29
+double8B

	full_text


double %48
Gphi8B>
<
	full_text/
-
+%39 = phi double [ %27, %20 ], [ %44, %35 ]
+double8B

	full_text


double %27
+double8B

	full_text


double %44
Dphi8B;
9
	full_text,
*
(%40 = phi i64 [ %25, %20 ], [ %42, %35 ]
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %42
Pphi8BG
E
	full_text8
6
4%41 = phi double [ 0.000000e+00, %20 ], [ %58, %35 ]
+double8B

	full_text


double %58
4add8B+
)
	full_text

%42 = add nsw i64 %40, 1
%i648B

	full_text
	
i64 %40
vgetelementptr8Bc
a
	full_textT
R
P%43 = getelementptr inbounds [66 x double], [66 x double]* %17, i64 %22, i64 %42
;[66 x double]*8B%
#
	full_text

[66 x double]* %17
%i648B

	full_text
	
i64 %22
%i648B

	full_text
	
i64 %42
Nload8BD
B
	full_text5
3
1%44 = load double, double* %43, align 8, !tbaa !8
-double*8B

	full_text

double* %43
7fadd8B-
+
	full_text

%45 = fadd double %39, %44
+double8B

	full_text


double %39
+double8B

	full_text


double %44
7fadd8B-
+
	full_text

%46 = fadd double %45, %38
+double8B

	full_text


double %45
+double8B

	full_text


double %38
vgetelementptr8Bc
a
	full_textT
R
P%47 = getelementptr inbounds [66 x double], [66 x double]* %17, i64 %24, i64 %42
;[66 x double]*8B%
#
	full_text

[66 x double]* %17
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %42
Nload8BD
B
	full_text5
3
1%48 = load double, double* %47, align 8, !tbaa !8
-double*8B

	full_text

double* %47
7fadd8B-
+
	full_text

%49 = fadd double %46, %48
+double8B

	full_text


double %46
+double8B

	full_text


double %48
7fadd8B-
+
	full_text

%50 = fadd double %49, %37
+double8B

	full_text


double %49
+double8B

	full_text


double %37
vgetelementptr8Bc
a
	full_textT
R
P%51 = getelementptr inbounds [66 x double], [66 x double]* %18, i64 %22, i64 %42
;[66 x double]*8B%
#
	full_text

[66 x double]* %18
%i648B

	full_text
	
i64 %22
%i648B

	full_text
	
i64 %42
Nload8BD
B
	full_text5
3
1%52 = load double, double* %51, align 8, !tbaa !8
-double*8B

	full_text

double* %51
7fadd8B-
+
	full_text

%53 = fadd double %50, %52
+double8B

	full_text


double %50
+double8B

	full_text


double %52
7fadd8B-
+
	full_text

%54 = fadd double %53, %36
+double8B

	full_text


double %53
+double8B

	full_text


double %36
vgetelementptr8Bc
a
	full_textT
R
P%55 = getelementptr inbounds [66 x double], [66 x double]* %18, i64 %24, i64 %42
;[66 x double]*8B%
#
	full_text

[66 x double]* %18
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %42
Nload8BD
B
	full_text5
3
1%56 = load double, double* %55, align 8, !tbaa !8
-double*8B

	full_text

double* %55
7fadd8B-
+
	full_text

%57 = fadd double %54, %56
+double8B

	full_text


double %54
+double8B

	full_text


double %56
7fadd8B-
+
	full_text

%58 = fadd double %41, %57
+double8B

	full_text


double %41
+double8B

	full_text


double %57
7icmp8B-
+
	full_text

%59 = icmp eq i64 %42, %34
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %34
:br8B2
0
	full_text#
!
br i1 %59, label %60, label %35
#i18B

	full_text


i1 %59
fphi8B]
[
	full_textN
L
J%61 = phi double [ 0.000000e+00, %8 ], [ 0.000000e+00, %16 ], [ %58, %35 ]
+double8B

	full_text


double %58
1shl8B(
&
	full_text

%62 = shl i64 %13, 32
%i648B

	full_text
	
i64 %13
9ashr8B/
-
	full_text 

%63 = ashr exact i64 %62, 32
%i648B

	full_text
	
i64 %62
^getelementptr8BK
I
	full_text<
:
8%64 = getelementptr inbounds double, double* %3, i64 %63
%i648B

	full_text
	
i64 %63
Nstore8BC
A
	full_text4
2
0store double %61, double* %64, align 8, !tbaa !8
+double8B

	full_text


double %61
-double*8B

	full_text

double* %64
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
5icmp8B+
)
	full_text

%65 = icmp eq i32 %14, 0
%i328B

	full_text
	
i32 %14
;br8B3
1
	full_text$
"
 br i1 %65, label %66, label %134
#i18B

	full_text


i1 %65
Ocall8BE
C
	full_text6
4
2%67 = tail call i64 @_Z14get_local_sizej(i32 0) #3
6icmp8B,
*
	full_text

%68 = icmp ugt i64 %67, 1
%i648B

	full_text
	
i64 %67
;br8B3
1
	full_text$
"
 br i1 %68, label %69, label %130
#i18B

	full_text


i1 %68
1add8B(
&
	full_text

%70 = add i64 %67, -1
%i648B

	full_text
	
i64 %67
1add8B(
&
	full_text

%71 = add i64 %67, -2
%i648B

	full_text
	
i64 %67
0and8B'
%
	full_text

%72 = and i64 %70, 7
%i648B

	full_text
	
i64 %70
6icmp8B,
*
	full_text

%73 = icmp ult i64 %71, 7
%i648B

	full_text
	
i64 %71
;br8B3
1
	full_text$
"
 br i1 %73, label %114, label %74
#i18B

	full_text


i1 %73
2sub8B)
'
	full_text

%75 = sub i64 %70, %72
%i648B

	full_text
	
i64 %70
%i648B

	full_text
	
i64 %72
'br8B

	full_text

br label %76
Cphi8B:
8
	full_text+
)
'%77 = phi i64 [ 1, %74 ], [ %111, %76 ]
&i648B

	full_text


i64 %111
Hphi8B?
=
	full_text0
.
,%78 = phi double [ %61, %74 ], [ %110, %76 ]
+double8B

	full_text


double %61
,double8B

	full_text

double %110
Ephi8B<
:
	full_text-
+
)%79 = phi i64 [ %75, %74 ], [ %112, %76 ]
%i648B

	full_text
	
i64 %75
&i648B

	full_text


i64 %112
^getelementptr8BK
I
	full_text<
:
8%80 = getelementptr inbounds double, double* %3, i64 %77
%i648B

	full_text
	
i64 %77
Nload8BD
B
	full_text5
3
1%81 = load double, double* %80, align 8, !tbaa !8
-double*8B

	full_text

double* %80
7fadd8B-
+
	full_text

%82 = fadd double %78, %81
+double8B

	full_text


double %78
+double8B

	full_text


double %81
8add8B/
-
	full_text 

%83 = add nuw nsw i64 %77, 1
%i648B

	full_text
	
i64 %77
^getelementptr8BK
I
	full_text<
:
8%84 = getelementptr inbounds double, double* %3, i64 %83
%i648B

	full_text
	
i64 %83
Nload8BD
B
	full_text5
3
1%85 = load double, double* %84, align 8, !tbaa !8
-double*8B

	full_text

double* %84
7fadd8B-
+
	full_text

%86 = fadd double %82, %85
+double8B

	full_text


double %82
+double8B

	full_text


double %85
4add8B+
)
	full_text

%87 = add nsw i64 %77, 2
%i648B

	full_text
	
i64 %77
^getelementptr8BK
I
	full_text<
:
8%88 = getelementptr inbounds double, double* %3, i64 %87
%i648B

	full_text
	
i64 %87
Nload8BD
B
	full_text5
3
1%89 = load double, double* %88, align 8, !tbaa !8
-double*8B

	full_text

double* %88
7fadd8B-
+
	full_text

%90 = fadd double %86, %89
+double8B

	full_text


double %86
+double8B

	full_text


double %89
4add8B+
)
	full_text

%91 = add nsw i64 %77, 3
%i648B

	full_text
	
i64 %77
^getelementptr8BK
I
	full_text<
:
8%92 = getelementptr inbounds double, double* %3, i64 %91
%i648B

	full_text
	
i64 %91
Nload8BD
B
	full_text5
3
1%93 = load double, double* %92, align 8, !tbaa !8
-double*8B

	full_text

double* %92
7fadd8B-
+
	full_text

%94 = fadd double %90, %93
+double8B

	full_text


double %90
+double8B

	full_text


double %93
4add8B+
)
	full_text

%95 = add nsw i64 %77, 4
%i648B

	full_text
	
i64 %77
^getelementptr8BK
I
	full_text<
:
8%96 = getelementptr inbounds double, double* %3, i64 %95
%i648B

	full_text
	
i64 %95
Nload8BD
B
	full_text5
3
1%97 = load double, double* %96, align 8, !tbaa !8
-double*8B

	full_text

double* %96
7fadd8B-
+
	full_text

%98 = fadd double %94, %97
+double8B

	full_text


double %94
+double8B

	full_text


double %97
4add8B+
)
	full_text

%99 = add nsw i64 %77, 5
%i648B

	full_text
	
i64 %77
_getelementptr8BL
J
	full_text=
;
9%100 = getelementptr inbounds double, double* %3, i64 %99
%i648B

	full_text
	
i64 %99
Pload8BF
D
	full_text7
5
3%101 = load double, double* %100, align 8, !tbaa !8
.double*8B

	full_text

double* %100
9fadd8B/
-
	full_text 

%102 = fadd double %98, %101
+double8B

	full_text


double %98
,double8B

	full_text

double %101
5add8B,
*
	full_text

%103 = add nsw i64 %77, 6
%i648B

	full_text
	
i64 %77
`getelementptr8BM
K
	full_text>
<
:%104 = getelementptr inbounds double, double* %3, i64 %103
&i648B

	full_text


i64 %103
Pload8BF
D
	full_text7
5
3%105 = load double, double* %104, align 8, !tbaa !8
.double*8B

	full_text

double* %104
:fadd8B0
.
	full_text!

%106 = fadd double %102, %105
,double8B

	full_text

double %102
,double8B

	full_text

double %105
5add8B,
*
	full_text

%107 = add nsw i64 %77, 7
%i648B

	full_text
	
i64 %77
`getelementptr8BM
K
	full_text>
<
:%108 = getelementptr inbounds double, double* %3, i64 %107
&i648B

	full_text


i64 %107
Pload8BF
D
	full_text7
5
3%109 = load double, double* %108, align 8, !tbaa !8
.double*8B

	full_text

double* %108
:fadd8B0
.
	full_text!

%110 = fadd double %106, %109
,double8B

	full_text

double %106
,double8B

	full_text

double %109
5add8B,
*
	full_text

%111 = add nsw i64 %77, 8
%i648B

	full_text
	
i64 %77
2add8B)
'
	full_text

%112 = add i64 %79, -8
%i648B

	full_text
	
i64 %79
7icmp8B-
+
	full_text

%113 = icmp eq i64 %112, 0
&i648B

	full_text


i64 %112
<br8B4
2
	full_text%
#
!br i1 %113, label %114, label %76
$i18B

	full_text
	
i1 %113
Kphi8	BB
@
	full_text3
1
/%115 = phi double [ undef, %69 ], [ %110, %76 ]
,double8	B

	full_text

double %110
Dphi8	B;
9
	full_text,
*
(%116 = phi i64 [ 1, %69 ], [ %111, %76 ]
&i648	B

	full_text


i64 %111
Iphi8	B@
>
	full_text1
/
-%117 = phi double [ %61, %69 ], [ %110, %76 ]
+double8	B

	full_text


double %61
,double8	B

	full_text

double %110
6icmp8	B,
*
	full_text

%118 = icmp eq i64 %72, 0
%i648	B

	full_text
	
i64 %72
=br8	B5
3
	full_text&
$
"br i1 %118, label %130, label %119
$i18	B

	full_text
	
i1 %118
(br8
B 

	full_text

br label %120
Iphi8B@
>
	full_text1
/
-%121 = phi i64 [ %127, %120 ], [ %116, %119 ]
&i648B

	full_text


i64 %127
&i648B

	full_text


i64 %116
Lphi8BC
A
	full_text4
2
0%122 = phi double [ %126, %120 ], [ %117, %119 ]
,double8B

	full_text

double %126
,double8B

	full_text

double %117
Hphi8B?
=
	full_text0
.
,%123 = phi i64 [ %128, %120 ], [ %72, %119 ]
&i648B

	full_text


i64 %128
%i648B

	full_text
	
i64 %72
`getelementptr8BM
K
	full_text>
<
:%124 = getelementptr inbounds double, double* %3, i64 %121
&i648B

	full_text


i64 %121
Pload8BF
D
	full_text7
5
3%125 = load double, double* %124, align 8, !tbaa !8
.double*8B

	full_text

double* %124
:fadd8B0
.
	full_text!

%126 = fadd double %122, %125
,double8B

	full_text

double %122
,double8B

	full_text

double %125
:add8B1
/
	full_text"
 
%127 = add nuw nsw i64 %121, 1
&i648B

	full_text


i64 %121
3add8B*
(
	full_text

%128 = add i64 %123, -1
&i648B

	full_text


i64 %123
7icmp8B-
+
	full_text

%129 = icmp eq i64 %128, 0
&i648B

	full_text


i64 %128
Mbr8BE
C
	full_text6
4
2br i1 %129, label %130, label %120, !llvm.loop !12
$i18B

	full_text
	
i1 %129
Zphi8BQ
O
	full_textB
@
>%131 = phi double [ %61, %66 ], [ %115, %114 ], [ %126, %120 ]
+double8B

	full_text


double %61
,double8B

	full_text

double %115
,double8B

	full_text

double %126
Ncall8BD
B
	full_text5
3
1%132 = tail call i64 @_Z12get_group_idj(i32 0) #3
`getelementptr8BM
K
	full_text>
<
:%133 = getelementptr inbounds double, double* %2, i64 %132
&i648B

	full_text


i64 %132
Pstore8BE
C
	full_text6
4
2store double %131, double* %133, align 8, !tbaa !8
,double8B

	full_text

double %131
.double*8B

	full_text

double* %133
(br8B 

	full_text

br label %134
$ret8B

	full_text


ret void
,double*8B

	full_text


double* %1
$i328B

	full_text


i32 %6
$i328B

	full_text


i32 %7
$i328B

	full_text


i32 %4
,double*8B

	full_text


double* %0
,double*8B

	full_text


double* %2
,double*8B

	full_text


double* %3
$i328B
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
$i648B

	full_text


i64 -8
-double8B

	full_text

double undef
4double8B&
$
	full_text

double 0.000000e+00
#i648B

	full_text	

i64 8
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 4
#i648B

	full_text	

i64 1
#i328B

	full_text	

i32 1
,i648B!

	full_text

i64 4294967296
#i648B

	full_text	

i64 5
#i648B

	full_text	

i64 6
#i648B

	full_text	

i64 7
$i648B

	full_text


i64 32
$i648B

	full_text


i64 -1
$i648B

	full_text


i64 -2
#i648B

	full_text	

i64 2
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 3        	
 		                      !" !! #$ #% #& ## '( '' )* )+ ), )) -. -- /0 /1 /2 // 34 33 55 68 79 77 :; :< :: => =? == @A @B @@ CD CE CC FG FF HI HH JK JL JM JJ NO NN PQ PR PP ST SU SS VW VX VY VV Z[ ZZ \] \^ \\ _` _a __ bc bd be bb fg ff hi hj hh kl km kk no np nq nn rs rr tu tv tt wx wy ww z{ z| zz }~ }	Ä  ÅÇ ÅÅ ÉÑ ÉÉ Ö
Ü ÖÖ áà á
â áá ää ãå ãã çé çè êë êê íì íï îî ñó ññ òô òò öõ öö úù úü û
† ûû °
£ ¢¢ §• §
¶ §§ ß® ß
© ßß ™
´ ™™ ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±± ≥
¥ ≥≥ µ∂ µµ ∑∏ ∑
π ∑∑ ∫ª ∫∫ º
Ω ºº æø ææ ¿¡ ¿
¬ ¿¿ √ƒ √√ ≈
∆ ≈≈ «» «« …  …
À …… ÃÕ ÃÃ Œ
œ ŒŒ –— –– “” “
‘ ““ ’÷ ’’ ◊
ÿ ◊◊ Ÿ⁄ ŸŸ €‹ €
› €€ ﬁﬂ ﬁﬁ ‡
· ‡‡ ‚„ ‚‚ ‰Â ‰
Ê ‰‰ ÁË ÁÁ È
Í ÈÈ ÎÏ ÎÎ ÌÓ Ì
Ô ÌÌ Ò  ÚÛ ÚÚ Ùı ÙÙ ˆ˜ ˆ
˘ ¯¯ ˙
˚ ˙˙ ¸˝ ¸
˛ ¸¸ ˇÄ ˇˇ ÅÇ ÅÖ Ñ
Ü ÑÑ áà á
â áá äã ä
å ää ç
é çç èê èè ëí ë
ì ëë îï îî ñó ññ òô òò öõ öù ú
û ú
ü úú †† °
¢ °° £§ £
• ££ ¶® © 	™ ´ ´ ¨ ≠ °Æ ÖÆ ™Æ ≥Æ ºÆ ≈Æ ŒÆ ◊Æ ‡Æ ÈÆ ç	Ø Ø 5    
            " $ % &# ( * + ,) . 0 1 2/ 43 8r 9- ;f <' >Z ?! AN B DH Ew GC I K LH MJ O@ QN RP T= U W XH YV [S ]Z ^\ `: a c dH eb g_ if jh l7 m o pH qn sk ur vF xt yH {5 |z ~w Ä ÇÅ ÑÉ Ü àÖ â	 åã éè ëê ìè ïè óî ôñ õö ùî üò † £ •Ì ¶û ®Ú ©¢ ´™ ≠§ Ø¨ ∞¢ ≤± ¥≥ ∂Æ ∏µ π¢ ª∫ Ωº ø∑ ¡æ ¬¢ ƒ√ ∆≈ »¿  « À¢ ÕÃ œŒ —… ”– ‘¢ ÷’ ÿ◊ ⁄“ ‹Ÿ ›¢ ﬂﬁ ·‡ „€ Â‚ Ê¢ ËÁ ÍÈ Ï‰ ÓÎ Ô¢ Òß ÛÚ ıÙ ˜Ì ˘ ˚ ˝Ì ˛ò Äˇ Çî Ö˙ Üë à¸ âñ ãò åÑ éç êá íè ìÑ ïä óñ ôò õ ù¯ ûë ü† ¢ú §° •    ç èç ß6 7í îí ú} } 7ú ¯ú û¶ ßÅ úÅ É° ¢É Ñˆ ¯ˆ ¢ö úö Ñ ±± ∞∞ ß ≤≤ ¥¥ ≥≥ ∞∞ † ¥¥ † ±± ä ≤≤ äè ≥≥ è
µ Ú∂ ¯∑ F∑ 	∑ 
∏ 
π Ù
π ˇ
π ò
∫ Ã	ª H
ª êª ¢
ª ±ª ˙
ª îº ä	Ω 
æ ’
ø ﬁ
¿ ò
¿ ö
¿ Á	¡ 	¡ 	¡ 
¡ Å
¡ É
¬ î
¬ ñ
√ ñ
ƒ ∫≈ ≈ 
≈ ã≈ è≈ †
∆ √"
pintgr_reduce"
_Z13get_global_idj"
_Z12get_local_idj"
_Z7barrierj"
_Z14get_local_sizej"
_Z12get_group_idj*ê
npb-LU-pintgr_reduce.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Ä

wgsize


wgsize_log1p
ΩaçA
 
transfer_bytes_log1p
ΩaçA

transfer_bytes
®ˇ»

devmap_label
 