

[external]
KcallBC
A
	full_text4
2
0%9 = tail call i64 @_Z13get_global_idj(i32 1) #4
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
0%11 = tail call i64 @_Z12get_group_idj(i32 0) #4
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
0%13 = tail call i64 @_Z12get_local_idj(i32 0) #4
6truncB-
+
	full_text

%14 = trunc i64 %13 to i32
#i64B

	full_text
	
i64 %13
3mulB,
*
	full_text

%15 = mul nsw i32 %10, %5
#i32B

	full_text
	
i32 %10
1addB*
(
	full_text

%16 = add nsw i32 %4, 1
0addB)
'
	full_text

%17 = add i32 %15, %12
#i32B

	full_text
	
i32 %15
#i32B

	full_text
	
i32 %12
0mulB)
'
	full_text

%18 = mul i32 %17, %16
#i32B

	full_text
	
i32 %17
#i32B

	full_text
	
i32 %16
5icmpB-
+
	full_text

%19 = icmp slt i32 %14, %4
#i32B

	full_text
	
i32 %14
8brB2
0
	full_text#
!
br i1 %19, label %20, label %47
!i1B

	full_text


i1 %19
1shl8B(
&
	full_text

%21 = shl i64 %13, 32
%i648B

	full_text
	
i64 %13
9ashr8B/
-
	full_text 

%22 = ashr exact i64 %21, 32
%i648B

	full_text
	
i64 %21
6sext8B,
*
	full_text

%23 = sext i32 %18 to i64
%i328B

	full_text
	
i32 %18
5sext8B+
)
	full_text

%24 = sext i32 %4 to i64
5add8B,
*
	full_text

%25 = add nsw i64 %24, -1
%i648B

	full_text
	
i64 %24
6sub8B-
+
	full_text

%26 = sub nsw i64 %25, %22
%i648B

	full_text
	
i64 %25
%i648B

	full_text
	
i64 %22
2lshr8B(
&
	full_text

%27 = lshr i64 %26, 6
%i648B

	full_text
	
i64 %26
0and8B'
%
	full_text

%28 = and i64 %27, 1
%i648B

	full_text
	
i64 %27
5icmp8B+
)
	full_text

%29 = icmp eq i64 %28, 0
%i648B

	full_text
	
i64 %28
:br8B2
0
	full_text#
!
br i1 %29, label %30, label %43
#i18B

	full_text


i1 %29
6add8B-
+
	full_text

%31 = add nsw i64 %22, %23
%i648B

	full_text
	
i64 %22
%i648B

	full_text
	
i64 %23
rgetelementptr8B_
]
	full_textP
N
L%32 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %0, i64 %31
%i648B

	full_text
	
i64 %31
Kbitcast8B>
<
	full_text/
-
+%33 = bitcast %struct.dcomplex* %32 to i64*
-struct*8B

	full_text

struct* %32
Hload8B>
<
	full_text/
-
+%34 = load i64, i64* %33, align 8, !tbaa !8
'i64*8B

	full_text


i64* %33
ígetelementptr8B
}
	full_textp
n
l%35 = getelementptr inbounds [128 x %struct.dcomplex], [128 x %struct.dcomplex]* @cffts1.ty1, i64 0, i64 %22
%i648B

	full_text
	
i64 %22
Kbitcast8B>
<
	full_text/
-
+%36 = bitcast %struct.dcomplex* %35 to i64*
-struct*8B

	full_text

struct* %35
Istore8B>
<
	full_text/
-
+store i64 %34, i64* %36, align 16, !tbaa !8
%i648B

	full_text
	
i64 %34
'i64*8B

	full_text


i64* %36
ygetelementptr8Bf
d
	full_textW
U
S%37 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %0, i64 %31, i32 1
%i648B

	full_text
	
i64 %31
Abitcast8B4
2
	full_text%
#
!%38 = bitcast double* %37 to i64*
-double*8B

	full_text

double* %37
Iload8B?
=
	full_text0
.
,%39 = load i64, i64* %38, align 8, !tbaa !13
'i64*8B

	full_text


i64* %38
õgetelementptr8Bá
Ñ
	full_textw
u
s%40 = getelementptr inbounds [128 x %struct.dcomplex], [128 x %struct.dcomplex]* @cffts1.ty1, i64 0, i64 %22, i32 1
%i648B

	full_text
	
i64 %22
Abitcast8B4
2
	full_text%
#
!%41 = bitcast double* %40 to i64*
-double*8B

	full_text

double* %40
Istore8B>
<
	full_text/
-
+store i64 %39, i64* %41, align 8, !tbaa !13
%i648B

	full_text
	
i64 %39
'i64*8B

	full_text


i64* %41
5add8B,
*
	full_text

%42 = add nsw i64 %22, 64
%i648B

	full_text
	
i64 %22
'br8B

	full_text

br label %43
Dphi8B;
9
	full_text,
*
(%44 = phi i64 [ %22, %20 ], [ %42, %30 ]
%i648B

	full_text
	
i64 %22
%i648B

	full_text
	
i64 %42
5icmp8B+
)
	full_text

%45 = icmp eq i64 %27, 0
%i648B

	full_text
	
i64 %27
:br8B2
0
	full_text#
!
br i1 %45, label %47, label %46
#i18B

	full_text


i1 %45
'br8B

	full_text

br label %79
Rphi8BI
G
	full_text:
8
6%48 = phi i1 [ false, %8 ], [ %19, %79 ], [ %19, %43 ]
#i18B

	full_text


i1 %19
#i18B

	full_text


i1 %19
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
‹call8B—
Œ
	full_text¿
Ω
∫tail call void @cfftz(i32 %3, i32 %7, i32 %4, %struct.dcomplex* %2, %struct.dcomplex* getelementptr inbounds ([128 x %struct.dcomplex], [128 x %struct.dcomplex]* @cffts1.ty1, i64 0, i64 0), %struct.dcomplex* getelementptr inbounds ([128 x %struct.dcomplex], [128 x %struct.dcomplex]* @cffts1.ty2, i64 0, i64 0)) #6
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
;br8B3
1
	full_text$
"
 br i1 %48, label %49, label %106
#i18B

	full_text


i1 %48
1shl8B(
&
	full_text

%50 = shl i64 %13, 32
%i648B

	full_text
	
i64 %13
9ashr8B/
-
	full_text 

%51 = ashr exact i64 %50, 32
%i648B

	full_text
	
i64 %50
6sext8B,
*
	full_text

%52 = sext i32 %18 to i64
%i328B

	full_text
	
i32 %18
5sext8B+
)
	full_text

%53 = sext i32 %4 to i64
5add8B,
*
	full_text

%54 = add nsw i64 %51, 64
%i648B

	full_text
	
i64 %51
8icmp8B.
,
	full_text

%55 = icmp sgt i64 %54, %53
%i648B

	full_text
	
i64 %54
%i648B

	full_text
	
i64 %53
Dselect8B8
6
	full_text)
'
%%56 = select i1 %55, i64 %54, i64 %53
#i18B

	full_text


i1 %55
%i648B

	full_text
	
i64 %54
%i648B

	full_text
	
i64 %53
5add8B,
*
	full_text

%57 = add nsw i64 %56, -1
%i648B

	full_text
	
i64 %56
6sub8B-
+
	full_text

%58 = sub nsw i64 %57, %51
%i648B

	full_text
	
i64 %57
%i648B

	full_text
	
i64 %51
2lshr8B(
&
	full_text

%59 = lshr i64 %58, 6
%i648B

	full_text
	
i64 %58
0and8B'
%
	full_text

%60 = and i64 %59, 1
%i648B

	full_text
	
i64 %59
5icmp8B+
)
	full_text

%61 = icmp eq i64 %60, 0
%i648B

	full_text
	
i64 %60
:br8B2
0
	full_text#
!
br i1 %61, label %62, label %75
#i18B

	full_text


i1 %61
6add8B-
+
	full_text

%63 = add nsw i64 %51, %52
%i648B

	full_text
	
i64 %51
%i648B

	full_text
	
i64 %52
ígetelementptr8B
}
	full_textp
n
l%64 = getelementptr inbounds [128 x %struct.dcomplex], [128 x %struct.dcomplex]* @cffts1.ty1, i64 0, i64 %51
%i648B

	full_text
	
i64 %51
Kbitcast8B>
<
	full_text/
-
+%65 = bitcast %struct.dcomplex* %64 to i64*
-struct*8B

	full_text

struct* %64
Iload8B?
=
	full_text0
.
,%66 = load i64, i64* %65, align 16, !tbaa !8
'i64*8B

	full_text


i64* %65
rgetelementptr8B_
]
	full_textP
N
L%67 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %1, i64 %63
%i648B

	full_text
	
i64 %63
Kbitcast8B>
<
	full_text/
-
+%68 = bitcast %struct.dcomplex* %67 to i64*
-struct*8B

	full_text

struct* %67
Hstore8B=
;
	full_text.
,
*store i64 %66, i64* %68, align 8, !tbaa !8
%i648B

	full_text
	
i64 %66
'i64*8B

	full_text


i64* %68
õgetelementptr8Bá
Ñ
	full_textw
u
s%69 = getelementptr inbounds [128 x %struct.dcomplex], [128 x %struct.dcomplex]* @cffts1.ty1, i64 0, i64 %51, i32 1
%i648B

	full_text
	
i64 %51
Abitcast8B4
2
	full_text%
#
!%70 = bitcast double* %69 to i64*
-double*8B

	full_text

double* %69
Iload8B?
=
	full_text0
.
,%71 = load i64, i64* %70, align 8, !tbaa !13
'i64*8B

	full_text


i64* %70
ygetelementptr8Bf
d
	full_textW
U
S%72 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %1, i64 %63, i32 1
%i648B

	full_text
	
i64 %63
Abitcast8B4
2
	full_text%
#
!%73 = bitcast double* %72 to i64*
-double*8B

	full_text

double* %72
Istore8B>
<
	full_text/
-
+store i64 %71, i64* %73, align 8, !tbaa !13
%i648B

	full_text
	
i64 %71
'i64*8B

	full_text


i64* %73
5add8B,
*
	full_text

%74 = add nsw i64 %51, 64
%i648B

	full_text
	
i64 %51
'br8B

	full_text

br label %75
Dphi8B;
9
	full_text,
*
(%76 = phi i64 [ %51, %49 ], [ %74, %62 ]
%i648B

	full_text
	
i64 %51
%i648B

	full_text
	
i64 %74
5icmp8B+
)
	full_text

%77 = icmp eq i64 %59, 0
%i648B

	full_text
	
i64 %59
;br8B3
1
	full_text$
"
 br i1 %77, label %106, label %78
#i18B

	full_text


i1 %77
(br8	B 

	full_text

br label %107
Ephi8
B<
:
	full_text-
+
)%80 = phi i64 [ %44, %46 ], [ %104, %79 ]
%i648
B

	full_text
	
i64 %44
&i648
B

	full_text


i64 %104
6add8
B-
+
	full_text

%81 = add nsw i64 %80, %23
%i648
B

	full_text
	
i64 %80
%i648
B

	full_text
	
i64 %23
rgetelementptr8
B_
]
	full_textP
N
L%82 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %0, i64 %81
%i648
B

	full_text
	
i64 %81
Kbitcast8
B>
<
	full_text/
-
+%83 = bitcast %struct.dcomplex* %82 to i64*
-struct*8
B

	full_text

struct* %82
Hload8
B>
<
	full_text/
-
+%84 = load i64, i64* %83, align 8, !tbaa !8
'i64*8
B

	full_text


i64* %83
ígetelementptr8
B
}
	full_textp
n
l%85 = getelementptr inbounds [128 x %struct.dcomplex], [128 x %struct.dcomplex]* @cffts1.ty1, i64 0, i64 %80
%i648
B

	full_text
	
i64 %80
Kbitcast8
B>
<
	full_text/
-
+%86 = bitcast %struct.dcomplex* %85 to i64*
-struct*8
B

	full_text

struct* %85
Istore8
B>
<
	full_text/
-
+store i64 %84, i64* %86, align 16, !tbaa !8
%i648
B

	full_text
	
i64 %84
'i64*8
B

	full_text


i64* %86
ygetelementptr8
Bf
d
	full_textW
U
S%87 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %0, i64 %81, i32 1
%i648
B

	full_text
	
i64 %81
Abitcast8
B4
2
	full_text%
#
!%88 = bitcast double* %87 to i64*
-double*8
B

	full_text

double* %87
Iload8
B?
=
	full_text0
.
,%89 = load i64, i64* %88, align 8, !tbaa !13
'i64*8
B

	full_text


i64* %88
õgetelementptr8
Bá
Ñ
	full_textw
u
s%90 = getelementptr inbounds [128 x %struct.dcomplex], [128 x %struct.dcomplex]* @cffts1.ty1, i64 0, i64 %80, i32 1
%i648
B

	full_text
	
i64 %80
Abitcast8
B4
2
	full_text%
#
!%91 = bitcast double* %90 to i64*
-double*8
B

	full_text

double* %90
Istore8
B>
<
	full_text/
-
+store i64 %89, i64* %91, align 8, !tbaa !13
%i648
B

	full_text
	
i64 %89
'i64*8
B

	full_text


i64* %91
5add8
B,
*
	full_text

%92 = add nsw i64 %80, 64
%i648
B

	full_text
	
i64 %80
6add8
B-
+
	full_text

%93 = add nsw i64 %92, %23
%i648
B

	full_text
	
i64 %92
%i648
B

	full_text
	
i64 %23
rgetelementptr8
B_
]
	full_textP
N
L%94 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %0, i64 %93
%i648
B

	full_text
	
i64 %93
Kbitcast8
B>
<
	full_text/
-
+%95 = bitcast %struct.dcomplex* %94 to i64*
-struct*8
B

	full_text

struct* %94
Hload8
B>
<
	full_text/
-
+%96 = load i64, i64* %95, align 8, !tbaa !8
'i64*8
B

	full_text


i64* %95
ígetelementptr8
B
}
	full_textp
n
l%97 = getelementptr inbounds [128 x %struct.dcomplex], [128 x %struct.dcomplex]* @cffts1.ty1, i64 0, i64 %92
%i648
B

	full_text
	
i64 %92
Kbitcast8
B>
<
	full_text/
-
+%98 = bitcast %struct.dcomplex* %97 to i64*
-struct*8
B

	full_text

struct* %97
Istore8
B>
<
	full_text/
-
+store i64 %96, i64* %98, align 16, !tbaa !8
%i648
B

	full_text
	
i64 %96
'i64*8
B

	full_text


i64* %98
ygetelementptr8
Bf
d
	full_textW
U
S%99 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %0, i64 %93, i32 1
%i648
B

	full_text
	
i64 %93
Bbitcast8
B5
3
	full_text&
$
"%100 = bitcast double* %99 to i64*
-double*8
B

	full_text

double* %99
Kload8
BA
?
	full_text2
0
.%101 = load i64, i64* %100, align 8, !tbaa !13
(i64*8
B

	full_text

	i64* %100
úgetelementptr8
Bà
Ö
	full_textx
v
t%102 = getelementptr inbounds [128 x %struct.dcomplex], [128 x %struct.dcomplex]* @cffts1.ty1, i64 0, i64 %92, i32 1
%i648
B

	full_text
	
i64 %92
Cbitcast8
B6
4
	full_text'
%
#%103 = bitcast double* %102 to i64*
.double*8
B

	full_text

double* %102
Kstore8
B@
>
	full_text1
/
-store i64 %101, i64* %103, align 8, !tbaa !13
&i648
B

	full_text


i64 %101
(i64*8
B

	full_text

	i64* %103
7add8
B.
,
	full_text

%104 = add nsw i64 %80, 128
%i648
B

	full_text
	
i64 %80
:icmp8
B0
.
	full_text!

%105 = icmp slt i64 %104, %24
&i648
B

	full_text


i64 %104
%i648
B

	full_text
	
i64 %24
;br8
B3
1
	full_text$
"
 br i1 %105, label %79, label %47
$i18
B

	full_text
	
i1 %105
$ret8B

	full_text


ret void
Gphi8B>
<
	full_text/
-
+%108 = phi i64 [ %76, %78 ], [ %132, %107 ]
%i648B

	full_text
	
i64 %76
&i648B

	full_text


i64 %132
8add8B/
-
	full_text 

%109 = add nsw i64 %108, %52
&i648B

	full_text


i64 %108
%i648B

	full_text
	
i64 %52
ïgetelementptr8BÅ

	full_textr
p
n%110 = getelementptr inbounds [128 x %struct.dcomplex], [128 x %struct.dcomplex]* @cffts1.ty1, i64 0, i64 %108
&i648B

	full_text


i64 %108
Mbitcast8B@
>
	full_text1
/
-%111 = bitcast %struct.dcomplex* %110 to i64*
.struct*8B

	full_text

struct* %110
Kload8BA
?
	full_text2
0
.%112 = load i64, i64* %111, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %111
tgetelementptr8Ba
_
	full_textR
P
N%113 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %1, i64 %109
&i648B

	full_text


i64 %109
Mbitcast8B@
>
	full_text1
/
-%114 = bitcast %struct.dcomplex* %113 to i64*
.struct*8B

	full_text

struct* %113
Jstore8B?
=
	full_text0
.
,store i64 %112, i64* %114, align 8, !tbaa !8
&i648B

	full_text


i64 %112
(i64*8B

	full_text

	i64* %114
ùgetelementptr8Bâ
Ü
	full_texty
w
u%115 = getelementptr inbounds [128 x %struct.dcomplex], [128 x %struct.dcomplex]* @cffts1.ty1, i64 0, i64 %108, i32 1
&i648B

	full_text


i64 %108
Cbitcast8B6
4
	full_text'
%
#%116 = bitcast double* %115 to i64*
.double*8B

	full_text

double* %115
Kload8BA
?
	full_text2
0
.%117 = load i64, i64* %116, align 8, !tbaa !13
(i64*8B

	full_text

	i64* %116
{getelementptr8Bh
f
	full_textY
W
U%118 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %1, i64 %109, i32 1
&i648B

	full_text


i64 %109
Cbitcast8B6
4
	full_text'
%
#%119 = bitcast double* %118 to i64*
.double*8B

	full_text

double* %118
Kstore8B@
>
	full_text1
/
-store i64 %117, i64* %119, align 8, !tbaa !13
&i648B

	full_text


i64 %117
(i64*8B

	full_text

	i64* %119
7add8B.
,
	full_text

%120 = add nsw i64 %108, 64
&i648B

	full_text


i64 %108
8add8B/
-
	full_text 

%121 = add nsw i64 %120, %52
&i648B

	full_text


i64 %120
%i648B

	full_text
	
i64 %52
ïgetelementptr8BÅ

	full_textr
p
n%122 = getelementptr inbounds [128 x %struct.dcomplex], [128 x %struct.dcomplex]* @cffts1.ty1, i64 0, i64 %120
&i648B

	full_text


i64 %120
Mbitcast8B@
>
	full_text1
/
-%123 = bitcast %struct.dcomplex* %122 to i64*
.struct*8B

	full_text

struct* %122
Kload8BA
?
	full_text2
0
.%124 = load i64, i64* %123, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %123
tgetelementptr8Ba
_
	full_textR
P
N%125 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %1, i64 %121
&i648B

	full_text


i64 %121
Mbitcast8B@
>
	full_text1
/
-%126 = bitcast %struct.dcomplex* %125 to i64*
.struct*8B

	full_text

struct* %125
Jstore8B?
=
	full_text0
.
,store i64 %124, i64* %126, align 8, !tbaa !8
&i648B

	full_text


i64 %124
(i64*8B

	full_text

	i64* %126
ùgetelementptr8Bâ
Ü
	full_texty
w
u%127 = getelementptr inbounds [128 x %struct.dcomplex], [128 x %struct.dcomplex]* @cffts1.ty1, i64 0, i64 %120, i32 1
&i648B

	full_text


i64 %120
Cbitcast8B6
4
	full_text'
%
#%128 = bitcast double* %127 to i64*
.double*8B

	full_text

double* %127
Kload8BA
?
	full_text2
0
.%129 = load i64, i64* %128, align 8, !tbaa !13
(i64*8B

	full_text

	i64* %128
{getelementptr8Bh
f
	full_textY
W
U%130 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %1, i64 %121, i32 1
&i648B

	full_text


i64 %121
Cbitcast8B6
4
	full_text'
%
#%131 = bitcast double* %130 to i64*
.double*8B

	full_text

double* %130
Kstore8B@
>
	full_text1
/
-store i64 %129, i64* %131, align 8, !tbaa !13
&i648B

	full_text


i64 %129
(i64*8B

	full_text

	i64* %131
8add8B/
-
	full_text 

%132 = add nsw i64 %108, 128
&i648B

	full_text


i64 %108
:icmp8B0
.
	full_text!

%133 = icmp slt i64 %132, %53
&i648B

	full_text


i64 %132
%i648B

	full_text
	
i64 %53
=br8B5
3
	full_text&
$
"br i1 %133, label %107, label %106
$i18B

	full_text
	
i1 %133
6struct*8B'
%
	full_text

%struct.dcomplex* %0
$i328B

	full_text


i32 %5
$i328B

	full_text


i32 %7
6struct*8B'
%
	full_text

%struct.dcomplex* %2
6struct*8B'
%
	full_text

%struct.dcomplex* %1
$i328B

	full_text


i32 %3
$i328B
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
-; undefined function B

	full_text

 
#i328B

	full_text	

i32 1
%i648B

	full_text
	
i64 128
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 6
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 1
z[128 x %struct.dcomplex]*8BY
W
	full_textJ
H
F@cffts1.ty1 = internal global [128 x %struct.dcomplex] undef, align 16
$i648B

	full_text


i64 64
%i18B

	full_text


i1 false
#i648B

	full_text	

i64 0
$i648B

	full_text


i64 -1
ústruct*8Bå
â
	full_text|
z
x%struct.dcomplex* getelementptr inbounds ([128 x %struct.dcomplex], [128 x %struct.dcomplex]* @cffts1.ty1, i64 0, i64 0)
ústruct*8Bå
â
	full_text|
z
x%struct.dcomplex* getelementptr inbounds ([128 x %struct.dcomplex], [128 x %struct.dcomplex]* @cffts1.ty2, i64 0, i64 0)       	  
 

                     !  "    #$ ## %& %% '( '' )* ), +- ++ ./ .. 01 00 23 22 45 44 67 66 89 8: 88 ;< ;; => == ?@ ?? AB AA CD CC EF EG EE HI HH JL KM KK NO NN PQ PT SU SS VV WW XX YZ Y\ [[ ]^ ]] _` __ aa bc bb de df dd gh gi gj gg kl kk mn mo mm pq pp rs rr tu tt vw vy xz xx {| {{ }~ }} Ä  Å
Ç ÅÅ ÉÑ ÉÉ ÖÜ Ö
á ÖÖ à
â àà äã ää åç åå é
è éé êë êê íì í
î íí ïñ ïï óô ò
ö òò õú õõ ùû ù° †
¢ †† £§ £
• ££ ¶
ß ¶¶ ®© ®® ™´ ™™ ¨
≠ ¨¨ ÆØ ÆÆ ∞± ∞
≤ ∞∞ ≥
¥ ≥≥ µ∂ µµ ∑∏ ∑∑ π
∫ ππ ªº ªª Ωæ Ω
ø ΩΩ ¿¡ ¿¿ ¬√ ¬
ƒ ¬¬ ≈
∆ ≈≈ «» «« …  …… À
Ã ÀÀ ÕŒ ÕÕ œ– œ
— œœ “
” ““ ‘’ ‘‘ ÷◊ ÷÷ ÿ
Ÿ ÿÿ ⁄€ ⁄⁄ ‹› ‹
ﬁ ‹‹ ﬂ‡ ﬂﬂ ·‚ ·
„ ·· ‰Â ‰Ë Á
È ÁÁ ÍÎ Í
Ï ÍÍ Ì
Ó ÌÌ Ô ÔÔ ÒÚ ÒÒ Û
Ù ÛÛ ıˆ ıı ˜¯ ˜
˘ ˜˜ ˙
˚ ˙˙ ¸˝ ¸¸ ˛ˇ ˛˛ Ä
Å ÄÄ ÇÉ ÇÇ ÑÖ Ñ
Ü ÑÑ áà áá âä â
ã ââ å
ç åå éè éé êë êê í
ì íí îï îî ñó ñ
ò ññ ô
ö ôô õú õõ ùû ùù ü
† üü °¢ °° £§ £
• ££ ¶ß ¶¶ ®© ®
™ ®® ´¨ ´≠ .≠ ;≠ ¶≠ ≥≠ ≈≠ “	Æ 
	Ø W	∞ W± Å± é± Û± Ä± í± ü≤ W≥ 	≥ ≥ 	≥ W≥ a   	 
           ! "  $# &% (' * , -+ /. 10 3 54 72 96 :+ <; >= @ BA D? FC G I LH M# ON Q T US Z \[ ^ `] cb ea fd hb ia jg lk n] om qp sr ut w] y_ z] |{ ~} Äx ÇÅ Ñ ÜÉ á] âà ãä çx èé ëå ìê î] ñ] ôï öp úõ ûK °ﬂ ¢† § •£ ß¶ ©® ´† ≠¨ Ø™ ±Æ ≤£ ¥≥ ∂µ ∏† ∫π º∑ æª ø† ¡¿ √ ƒ¬ ∆≈ »«  ¿ ÃÀ Œ… –Õ —¬ ”“ ’‘ ◊¿ Ÿÿ €÷ ›⁄ ﬁ† ‡ﬂ ‚ „· Âò Ë¶ ÈÁ Î_ ÏÁ ÓÌ Ô ÚÍ ÙÛ ˆÒ ¯ı ˘Á ˚˙ ˝¸ ˇÍ ÅÄ É˛ ÖÇ ÜÁ àá ä_ ãá çå èé ëâ ìí ïê óî òá öô úõ ûâ †ü ¢ù §° •Á ß¶ ©a ™® ¨  S) +) KY [Y ÊJ KP SP Rv xv òR †ó òù Êù ü‰ †‰ Sü Á´ Á´ Ê ∑∑ ∏∏ ¥¥ µµ Ê ∂∂V ∑∑ V ¥¥  µµ  ∂∂ W ∏∏ WX ∑∑ Xπ 	π 	π ;	π Aπ Vπ X
π à
π é
π ≥
π π
π “
π ÿ
π ˙
π Ä
π ô
π ü
∫ ﬂ
∫ ¶	ª 	ª 	ª [	ª ]	º #	º pΩ Ω 	æ %	æ rø 4ø Aø {ø àø ¨ø πø Àø ÿø Ìø ˙ø åø ô	¿ H	¿ b
¿ ï
¿ ¿
¿ á¡ S	¬ '	¬ 4	¬ A	¬ N	¬ t	¬ {
¬ à
¬ õ
¬ ¨
¬ π
¬ À
¬ ÿ
¬ Ì
¬ ˙
¬ å
¬ ô	√ 	√ k	ƒ W	≈ W"
cffts1"
_Z13get_global_idj"
_Z12get_group_idj"
_Z12get_local_idj"
_Z7barrierj"
cfftz*â
npb-FT-cffts1.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Ä
 
transfer_bytes_log1p
ŒÓÜA

wgsize
@

wgsize_log1p
ŒÓÜA

transfer_bytes
êòä


devmap_label
