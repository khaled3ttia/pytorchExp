

[external]
JcallBB
@
	full_text3
1
/%4 = tail call i64 @_Z12get_local_idj(i32 0) #2
JcallBB
@
	full_text3
1
/%5 = tail call i64 @_Z12get_group_idj(i32 0) #2
,shlB%
#
	full_text

%6 = shl i64 %5, 9
"i64B

	full_text


i64 %5
-addB&
$
	full_text

%7 = add i64 %6, %4
"i64B

	full_text


i64 %6
"i64B

	full_text


i64 %4
-shlB&
$
	full_text

%8 = shl i64 %7, 32
"i64B

	full_text


i64 %7
5ashrB-
+
	full_text

%9 = ashr exact i64 %8, 32
"i64B

	full_text


i64 %8
egetelementptrBT
R
	full_textE
C
A%10 = getelementptr inbounds <2 x float>, <2 x float>* %0, i64 %9
"i64B

	full_text


i64 %9
VloadBN
L
	full_text?
=
;%11 = load <2 x float>, <2 x float>* %10, align 8, !tbaa !9
5<2 x float>*B#
!
	full_text

<2 x float>* %10
fgetelementptrBU
S
	full_textF
D
B%12 = getelementptr inbounds <2 x float>, <2 x float>* %10, i64 64
5<2 x float>*B#
!
	full_text

<2 x float>* %10
VloadBN
L
	full_text?
=
;%13 = load <2 x float>, <2 x float>* %12, align 8, !tbaa !9
5<2 x float>*B#
!
	full_text

<2 x float>* %12
ggetelementptrBV
T
	full_textG
E
C%14 = getelementptr inbounds <2 x float>, <2 x float>* %10, i64 128
5<2 x float>*B#
!
	full_text

<2 x float>* %10
VloadBN
L
	full_text?
=
;%15 = load <2 x float>, <2 x float>* %14, align 8, !tbaa !9
5<2 x float>*B#
!
	full_text

<2 x float>* %14
ggetelementptrBV
T
	full_textG
E
C%16 = getelementptr inbounds <2 x float>, <2 x float>* %10, i64 192
5<2 x float>*B#
!
	full_text

<2 x float>* %10
VloadBN
L
	full_text?
=
;%17 = load <2 x float>, <2 x float>* %16, align 8, !tbaa !9
5<2 x float>*B#
!
	full_text

<2 x float>* %16
ggetelementptrBV
T
	full_textG
E
C%18 = getelementptr inbounds <2 x float>, <2 x float>* %10, i64 256
5<2 x float>*B#
!
	full_text

<2 x float>* %10
VloadBN
L
	full_text?
=
;%19 = load <2 x float>, <2 x float>* %18, align 8, !tbaa !9
5<2 x float>*B#
!
	full_text

<2 x float>* %18
ggetelementptrBV
T
	full_textG
E
C%20 = getelementptr inbounds <2 x float>, <2 x float>* %10, i64 320
5<2 x float>*B#
!
	full_text

<2 x float>* %10
VloadBN
L
	full_text?
=
;%21 = load <2 x float>, <2 x float>* %20, align 8, !tbaa !9
5<2 x float>*B#
!
	full_text

<2 x float>* %20
ggetelementptrBV
T
	full_textG
E
C%22 = getelementptr inbounds <2 x float>, <2 x float>* %10, i64 384
5<2 x float>*B#
!
	full_text

<2 x float>* %10
VloadBN
L
	full_text?
=
;%23 = load <2 x float>, <2 x float>* %22, align 8, !tbaa !9
5<2 x float>*B#
!
	full_text

<2 x float>* %22
ggetelementptrBV
T
	full_textG
E
C%24 = getelementptr inbounds <2 x float>, <2 x float>* %10, i64 448
5<2 x float>*B#
!
	full_text

<2 x float>* %10
VloadBN
L
	full_text?
=
;%25 = load <2 x float>, <2 x float>* %24, align 8, !tbaa !9
5<2 x float>*B#
!
	full_text

<2 x float>* %24
3sextB+
)
	full_text

%26 = sext i32 %1 to i64
ggetelementptrBV
T
	full_textG
E
C%27 = getelementptr inbounds <2 x float>, <2 x float>* %10, i64 %26
5<2 x float>*B#
!
	full_text

<2 x float>* %10
#i64B

	full_text
	
i64 %26
VloadBN
L
	full_text?
=
;%28 = load <2 x float>, <2 x float>* %27, align 8, !tbaa !9
5<2 x float>*B#
!
	full_text

<2 x float>* %27
3addB,
*
	full_text

%29 = add nsw i64 %26, 64
#i64B

	full_text
	
i64 %26
ggetelementptrBV
T
	full_textG
E
C%30 = getelementptr inbounds <2 x float>, <2 x float>* %10, i64 %29
5<2 x float>*B#
!
	full_text

<2 x float>* %10
#i64B

	full_text
	
i64 %29
VloadBN
L
	full_text?
=
;%31 = load <2 x float>, <2 x float>* %30, align 8, !tbaa !9
5<2 x float>*B#
!
	full_text

<2 x float>* %30
4addB-
+
	full_text

%32 = add nsw i64 %26, 128
#i64B

	full_text
	
i64 %26
ggetelementptrBV
T
	full_textG
E
C%33 = getelementptr inbounds <2 x float>, <2 x float>* %10, i64 %32
5<2 x float>*B#
!
	full_text

<2 x float>* %10
#i64B

	full_text
	
i64 %32
VloadBN
L
	full_text?
=
;%34 = load <2 x float>, <2 x float>* %33, align 8, !tbaa !9
5<2 x float>*B#
!
	full_text

<2 x float>* %33
4addB-
+
	full_text

%35 = add nsw i64 %26, 192
#i64B

	full_text
	
i64 %26
ggetelementptrBV
T
	full_textG
E
C%36 = getelementptr inbounds <2 x float>, <2 x float>* %10, i64 %35
5<2 x float>*B#
!
	full_text

<2 x float>* %10
#i64B

	full_text
	
i64 %35
VloadBN
L
	full_text?
=
;%37 = load <2 x float>, <2 x float>* %36, align 8, !tbaa !9
5<2 x float>*B#
!
	full_text

<2 x float>* %36
4addB-
+
	full_text

%38 = add nsw i64 %26, 256
#i64B

	full_text
	
i64 %26
ggetelementptrBV
T
	full_textG
E
C%39 = getelementptr inbounds <2 x float>, <2 x float>* %10, i64 %38
5<2 x float>*B#
!
	full_text

<2 x float>* %10
#i64B

	full_text
	
i64 %38
VloadBN
L
	full_text?
=
;%40 = load <2 x float>, <2 x float>* %39, align 8, !tbaa !9
5<2 x float>*B#
!
	full_text

<2 x float>* %39
4addB-
+
	full_text

%41 = add nsw i64 %26, 320
#i64B

	full_text
	
i64 %26
ggetelementptrBV
T
	full_textG
E
C%42 = getelementptr inbounds <2 x float>, <2 x float>* %10, i64 %41
5<2 x float>*B#
!
	full_text

<2 x float>* %10
#i64B

	full_text
	
i64 %41
VloadBN
L
	full_text?
=
;%43 = load <2 x float>, <2 x float>* %42, align 8, !tbaa !9
5<2 x float>*B#
!
	full_text

<2 x float>* %42
4addB-
+
	full_text

%44 = add nsw i64 %26, 384
#i64B

	full_text
	
i64 %26
ggetelementptrBV
T
	full_textG
E
C%45 = getelementptr inbounds <2 x float>, <2 x float>* %10, i64 %44
5<2 x float>*B#
!
	full_text

<2 x float>* %10
#i64B

	full_text
	
i64 %44
VloadBN
L
	full_text?
=
;%46 = load <2 x float>, <2 x float>* %45, align 8, !tbaa !9
5<2 x float>*B#
!
	full_text

<2 x float>* %45
4addB-
+
	full_text

%47 = add nsw i64 %26, 448
#i64B

	full_text
	
i64 %26
ggetelementptrBV
T
	full_textG
E
C%48 = getelementptr inbounds <2 x float>, <2 x float>* %10, i64 %47
5<2 x float>*B#
!
	full_text

<2 x float>* %10
#i64B

	full_text
	
i64 %47
VloadBN
L
	full_text?
=
;%49 = load <2 x float>, <2 x float>* %48, align 8, !tbaa !9
5<2 x float>*B#
!
	full_text

<2 x float>* %48
PextractelementB>
<
	full_text/
-
+%50 = extractelement <2 x float> %11, i64 0
3<2 x float>B"
 
	full_text

<2 x float> %11
PextractelementB>
<
	full_text/
-
+%51 = extractelement <2 x float> %28, i64 0
3<2 x float>B"
 
	full_text

<2 x float> %28
8fcmpB0
.
	full_text!

%52 = fcmp une float %50, %51
'floatB

	full_text

	float %50
'floatB

	full_text

	float %51
8brB2
0
	full_text#
!
br i1 %52, label %57, label %53
!i1B

	full_text


i1 %52
Rextractelement8B>
<
	full_text/
-
+%54 = extractelement <2 x float> %11, i64 1
5<2 x float>8B"
 
	full_text

<2 x float> %11
Rextractelement8B>
<
	full_text/
-
+%55 = extractelement <2 x float> %28, i64 1
5<2 x float>8B"
 
	full_text

<2 x float> %28
:fcmp8B0
.
	full_text!

%56 = fcmp une float %54, %55
)float8B

	full_text

	float %54
)float8B

	full_text

	float %55
:br8B2
0
	full_text#
!
br i1 %56, label %57, label %58
#i18B

	full_text


i1 %56
Fstore8B;
9
	full_text,
*
(store i32 1, i32* %2, align 4, !tbaa !12
'br8B

	full_text

br label %58
Rextractelement8B>
<
	full_text/
-
+%59 = extractelement <2 x float> %13, i64 0
5<2 x float>8B"
 
	full_text

<2 x float> %13
Rextractelement8B>
<
	full_text/
-
+%60 = extractelement <2 x float> %31, i64 0
5<2 x float>8B"
 
	full_text

<2 x float> %31
:fcmp8B0
.
	full_text!

%61 = fcmp une float %59, %60
)float8B

	full_text

	float %59
)float8B

	full_text

	float %60
:br8B2
0
	full_text#
!
br i1 %61, label %66, label %62
#i18B

	full_text


i1 %61
Rextractelement8B>
<
	full_text/
-
+%63 = extractelement <2 x float> %13, i64 1
5<2 x float>8B"
 
	full_text

<2 x float> %13
Rextractelement8B>
<
	full_text/
-
+%64 = extractelement <2 x float> %31, i64 1
5<2 x float>8B"
 
	full_text

<2 x float> %31
:fcmp8B0
.
	full_text!

%65 = fcmp une float %63, %64
)float8B

	full_text

	float %63
)float8B

	full_text

	float %64
:br8B2
0
	full_text#
!
br i1 %65, label %66, label %67
#i18B

	full_text


i1 %65
Fstore8B;
9
	full_text,
*
(store i32 1, i32* %2, align 4, !tbaa !12
'br8B

	full_text

br label %67
Rextractelement8B>
<
	full_text/
-
+%68 = extractelement <2 x float> %15, i64 0
5<2 x float>8B"
 
	full_text

<2 x float> %15
Rextractelement8B>
<
	full_text/
-
+%69 = extractelement <2 x float> %34, i64 0
5<2 x float>8B"
 
	full_text

<2 x float> %34
:fcmp8B0
.
	full_text!

%70 = fcmp une float %68, %69
)float8B

	full_text

	float %68
)float8B

	full_text

	float %69
:br8B2
0
	full_text#
!
br i1 %70, label %75, label %71
#i18B

	full_text


i1 %70
Rextractelement8B>
<
	full_text/
-
+%72 = extractelement <2 x float> %15, i64 1
5<2 x float>8B"
 
	full_text

<2 x float> %15
Rextractelement8B>
<
	full_text/
-
+%73 = extractelement <2 x float> %34, i64 1
5<2 x float>8B"
 
	full_text

<2 x float> %34
:fcmp8B0
.
	full_text!

%74 = fcmp une float %72, %73
)float8B

	full_text

	float %72
)float8B

	full_text

	float %73
:br8B2
0
	full_text#
!
br i1 %74, label %75, label %76
#i18B

	full_text


i1 %74
Fstore8B;
9
	full_text,
*
(store i32 1, i32* %2, align 4, !tbaa !12
'br8B

	full_text

br label %76
Rextractelement8	B>
<
	full_text/
-
+%77 = extractelement <2 x float> %17, i64 0
5<2 x float>8	B"
 
	full_text

<2 x float> %17
Rextractelement8	B>
<
	full_text/
-
+%78 = extractelement <2 x float> %37, i64 0
5<2 x float>8	B"
 
	full_text

<2 x float> %37
:fcmp8	B0
.
	full_text!

%79 = fcmp une float %77, %78
)float8	B

	full_text

	float %77
)float8	B

	full_text

	float %78
:br8	B2
0
	full_text#
!
br i1 %79, label %84, label %80
#i18	B

	full_text


i1 %79
Rextractelement8
B>
<
	full_text/
-
+%81 = extractelement <2 x float> %17, i64 1
5<2 x float>8
B"
 
	full_text

<2 x float> %17
Rextractelement8
B>
<
	full_text/
-
+%82 = extractelement <2 x float> %37, i64 1
5<2 x float>8
B"
 
	full_text

<2 x float> %37
:fcmp8
B0
.
	full_text!

%83 = fcmp une float %81, %82
)float8
B

	full_text

	float %81
)float8
B

	full_text

	float %82
:br8
B2
0
	full_text#
!
br i1 %83, label %84, label %85
#i18
B

	full_text


i1 %83
Fstore8B;
9
	full_text,
*
(store i32 1, i32* %2, align 4, !tbaa !12
'br8B

	full_text

br label %85
Rextractelement8B>
<
	full_text/
-
+%86 = extractelement <2 x float> %19, i64 0
5<2 x float>8B"
 
	full_text

<2 x float> %19
Rextractelement8B>
<
	full_text/
-
+%87 = extractelement <2 x float> %40, i64 0
5<2 x float>8B"
 
	full_text

<2 x float> %40
:fcmp8B0
.
	full_text!

%88 = fcmp une float %86, %87
)float8B

	full_text

	float %86
)float8B

	full_text

	float %87
:br8B2
0
	full_text#
!
br i1 %88, label %93, label %89
#i18B

	full_text


i1 %88
Rextractelement8B>
<
	full_text/
-
+%90 = extractelement <2 x float> %19, i64 1
5<2 x float>8B"
 
	full_text

<2 x float> %19
Rextractelement8B>
<
	full_text/
-
+%91 = extractelement <2 x float> %40, i64 1
5<2 x float>8B"
 
	full_text

<2 x float> %40
:fcmp8B0
.
	full_text!

%92 = fcmp une float %90, %91
)float8B

	full_text

	float %90
)float8B

	full_text

	float %91
:br8B2
0
	full_text#
!
br i1 %92, label %93, label %94
#i18B

	full_text


i1 %92
Fstore8B;
9
	full_text,
*
(store i32 1, i32* %2, align 4, !tbaa !12
'br8B

	full_text

br label %94
Rextractelement8B>
<
	full_text/
-
+%95 = extractelement <2 x float> %21, i64 0
5<2 x float>8B"
 
	full_text

<2 x float> %21
Rextractelement8B>
<
	full_text/
-
+%96 = extractelement <2 x float> %43, i64 0
5<2 x float>8B"
 
	full_text

<2 x float> %43
:fcmp8B0
.
	full_text!

%97 = fcmp une float %95, %96
)float8B

	full_text

	float %95
)float8B

	full_text

	float %96
;br8B3
1
	full_text$
"
 br i1 %97, label %102, label %98
#i18B

	full_text


i1 %97
Rextractelement8B>
<
	full_text/
-
+%99 = extractelement <2 x float> %21, i64 1
5<2 x float>8B"
 
	full_text

<2 x float> %21
Sextractelement8B?
=
	full_text0
.
,%100 = extractelement <2 x float> %43, i64 1
5<2 x float>8B"
 
	full_text

<2 x float> %43
<fcmp8B2
0
	full_text#
!
%101 = fcmp une float %99, %100
)float8B

	full_text

	float %99
*float8B

	full_text


float %100
=br8B5
3
	full_text&
$
"br i1 %101, label %102, label %103
$i18B

	full_text
	
i1 %101
Fstore8B;
9
	full_text,
*
(store i32 1, i32* %2, align 4, !tbaa !12
(br8B 

	full_text

br label %103
Sextractelement8B?
=
	full_text0
.
,%104 = extractelement <2 x float> %23, i64 0
5<2 x float>8B"
 
	full_text

<2 x float> %23
Sextractelement8B?
=
	full_text0
.
,%105 = extractelement <2 x float> %46, i64 0
5<2 x float>8B"
 
	full_text

<2 x float> %46
=fcmp8B3
1
	full_text$
"
 %106 = fcmp une float %104, %105
*float8B

	full_text


float %104
*float8B

	full_text


float %105
=br8B5
3
	full_text&
$
"br i1 %106, label %111, label %107
$i18B

	full_text
	
i1 %106
Sextractelement8B?
=
	full_text0
.
,%108 = extractelement <2 x float> %23, i64 1
5<2 x float>8B"
 
	full_text

<2 x float> %23
Sextractelement8B?
=
	full_text0
.
,%109 = extractelement <2 x float> %46, i64 1
5<2 x float>8B"
 
	full_text

<2 x float> %46
=fcmp8B3
1
	full_text$
"
 %110 = fcmp une float %108, %109
*float8B

	full_text


float %108
*float8B

	full_text


float %109
=br8B5
3
	full_text&
$
"br i1 %110, label %111, label %112
$i18B

	full_text
	
i1 %110
Fstore8B;
9
	full_text,
*
(store i32 1, i32* %2, align 4, !tbaa !12
(br8B 

	full_text

br label %112
Sextractelement8B?
=
	full_text0
.
,%113 = extractelement <2 x float> %25, i64 0
5<2 x float>8B"
 
	full_text

<2 x float> %25
Sextractelement8B?
=
	full_text0
.
,%114 = extractelement <2 x float> %49, i64 0
5<2 x float>8B"
 
	full_text

<2 x float> %49
=fcmp8B3
1
	full_text$
"
 %115 = fcmp une float %113, %114
*float8B

	full_text


float %113
*float8B

	full_text


float %114
=br8B5
3
	full_text&
$
"br i1 %115, label %120, label %116
$i18B

	full_text
	
i1 %115
Sextractelement8B?
=
	full_text0
.
,%117 = extractelement <2 x float> %25, i64 1
5<2 x float>8B"
 
	full_text

<2 x float> %25
Sextractelement8B?
=
	full_text0
.
,%118 = extractelement <2 x float> %49, i64 1
5<2 x float>8B"
 
	full_text

<2 x float> %49
=fcmp8B3
1
	full_text$
"
 %119 = fcmp une float %117, %118
*float8B

	full_text


float %117
*float8B

	full_text


float %118
=br8B5
3
	full_text&
$
"br i1 %119, label %120, label %121
$i18B

	full_text
	
i1 %119
Fstore8B;
9
	full_text,
*
(store i32 1, i32* %2, align 4, !tbaa !12
(br8B 

	full_text

br label %121
$ret8B

	full_text


ret void
6<2 x float>*8B"
 
	full_text

<2 x float>* %0
$i328B

	full_text


i32 %1
&i32*8B

	full_text
	
i32* %2
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
%i648B

	full_text
	
i64 128
%i648B

	full_text
	
i64 192
$i648B

	full_text


i64 32
%i648B

	full_text
	
i64 256
#i328B

	full_text	

i32 0
%i648B

	full_text
	
i64 320
%i648B

	full_text
	
i64 384
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 1
%i648B

	full_text
	
i64 448
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 9
$i648B

	full_text


i64 64       	  
 

                      !    "# "" $% $$ &' && () (( *+ ** ,, -. -/ -- 01 00 23 22 45 46 44 78 77 9: 99 ;< ;= ;; >? >> @A @@ BC BD BB EF EE GH GG IJ IK II LM LL NO NN PQ PR PP ST SS UV UU WX WY WW Z[ ZZ \] \\ ^_ ^` ^^ ab aa cd cc ef ee gh gi gg jk jm ll no nn pq pr pp st su vx ww yz yy {| {} {{ ~ ~Å ÄÄ ÇÉ ÇÇ ÑÖ Ñ
Ü ÑÑ áà áâ äå ãã çé çç èê è
ë èè íì íï îî ñó ññ òô ò
ö òò õú õù û† üü °¢ °° £§ £
• ££ ¶ß ¶© ®® ™´ ™™ ¨≠ ¨
Æ ¨¨ Ø∞ Ø± ≤¥ ≥≥ µ∂ µµ ∑∏ ∑
π ∑∑ ∫ª ∫Ω ºº æø ææ ¿¡ ¿
¬ ¿¿ √ƒ √≈ ∆» «« …  …… ÀÃ À
Õ ÀÀ Œœ Œ— –– “” ““ ‘’ ‘
÷ ‘‘ ◊ÿ ◊Ÿ ⁄‹ €€ ›ﬁ ›› ﬂ‡ ﬂ
· ﬂﬂ ‚„ ‚Â ‰‰ ÊÁ ÊÊ ËÈ Ë
Í ËË ÎÏ ÎÌ Ó ÔÔ ÒÚ ÒÒ ÛÙ Û
ı ÛÛ ˆ˜ ˆ˘ ¯¯ ˙˚ ˙˙ ¸˝ ¸
˛ ¸¸ ˇÄ ˇÅ ÇÑ Ö ,	Ü u
Ü â
Ü ù
Ü ±
Ü ≈
Ü Ÿ
Ü Ì
Ü Å    	 
           !  # %$ ' )( + ., /- 1, 3 52 64 8, : <9 =; ?, A C@ DB F, H JG KI M, O QN RP T, V XU YW [, ] _\ `^ b d0 fc he ig k m0 ol qn rp t x7 zw |y }{  Å7 ÉÄ ÖÇ ÜÑ à å> éã êç ëè ì ï> óî ôñ öò ú †E ¢ü §° •£ ß ©E ´® ≠™ Æ¨ ∞ ¥L ∂≥ ∏µ π∑ ª ΩL øº ¡æ ¬¿ ƒ" »S  « Ã… ÕÀ œ" —S ”– ’“ ÷‘ ÿ& ‹Z ﬁ€ ‡› ·ﬂ „& ÂZ Á‰ ÈÊ ÍË Ï* a ÚÔ ÙÒ ıÛ ˜* ˘a ˚¯ ˝˙ ˛¸ Äj uj lv ws us w~ â~ Ää ãá âá ãí ùí îû üõ ùõ ü¶ ±¶ ®≤ ≥Ø ±Ø ≥∫ ≈∫ º∆ «√ ≈√ «Œ ŸŒ –⁄ €◊ Ÿ◊ €‚ Ì‚ ‰Ó ÔÎ ÌÎ Ôˆ Åˆ ¯Ç Éˇ Åˇ É áá É àà áá  àà 	â 	â 9	ä 	ä @	ã 	ã 
	å 	å Gç ç 	é  	é N	è $	è U	ê c	ê e	ê w	ê y
ê ã
ê ç
ê ü
ê °
ê ≥
ê µ
ê «
ê …
ê €
ê ›
ê Ô
ê Ò	ë l	ë n
ë Ä
ë Ç
ë î
ë ñ
ë ®
ë ™
ë º
ë æ
ë –
ë “
ë ‰
ë Ê
ë ¯
ë ˙	í (	í \ì uì âì ùì ±ì ≈ì Ÿì Ìì Å	î 	ï 	ï 2"
	chk1D_512"
_Z12get_local_idj"
_Z12get_group_idj*î
shoc-1.1.5-FFT-chk1D_512.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Ä
 
transfer_bytes_log1p
´yzA

wgsize
@

devmap_label
 

transfer_bytes
àÄÄ

wgsize_log1p
´yzA