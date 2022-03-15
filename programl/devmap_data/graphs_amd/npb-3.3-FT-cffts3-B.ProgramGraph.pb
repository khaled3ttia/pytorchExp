
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
1addB*
(
	full_text

%15 = add nsw i32 %4, 1
4mulB-
+
	full_text

%16 = mul nsw i32 %15, %10
#i32B

	full_text
	
i32 %15
#i32B

	full_text
	
i32 %10
4addB-
+
	full_text

%17 = add nsw i32 %16, %12
#i32B

	full_text
	
i32 %16
#i32B

	full_text
	
i32 %12
5icmpB-
+
	full_text

%18 = icmp slt i32 %14, %6
#i32B

	full_text
	
i32 %14
8brB2
0
	full_text#
!
br i1 %18, label %19, label %49
!i1B

	full_text


i1 %18
1mul8B(
&
	full_text

%20 = mul i32 %15, %5
%i328B

	full_text
	
i32 %15
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
5sext8B+
)
	full_text

%23 = sext i32 %6 to i64
5add8B,
*
	full_text

%24 = add nsw i64 %23, -1
%i648B

	full_text
	
i64 %23
6sub8B-
+
	full_text

%25 = sub nsw i64 %24, %22
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %22
2lshr8B(
&
	full_text

%26 = lshr i64 %25, 6
%i648B

	full_text
	
i64 %25
0and8B'
%
	full_text

%27 = and i64 %26, 1
%i648B

	full_text
	
i64 %26
5icmp8B+
)
	full_text

%28 = icmp eq i64 %27, 0
%i648B

	full_text
	
i64 %27
:br8B2
0
	full_text#
!
br i1 %28, label %29, label %45
#i18B

	full_text


i1 %28
8trunc8B-
+
	full_text

%30 = trunc i64 %22 to i32
%i648B

	full_text
	
i64 %22
2mul8B)
'
	full_text

%31 = mul i32 %20, %30
%i328B

	full_text
	
i32 %20
%i328B

	full_text
	
i32 %30
6add8B-
+
	full_text

%32 = add nsw i32 %31, %17
%i328B

	full_text
	
i32 %31
%i328B

	full_text
	
i32 %17
6sext8B,
*
	full_text

%33 = sext i32 %32 to i64
%i328B

	full_text
	
i32 %32
rgetelementptr8B_
]
	full_textP
N
L%34 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %0, i64 %33
%i648B

	full_text
	
i64 %33
Kbitcast8B>
<
	full_text/
-
+%35 = bitcast %struct.dcomplex* %34 to i64*
-struct*8B

	full_text

struct* %34
Hload8B>
<
	full_text/
-
+%36 = load i64, i64* %35, align 8, !tbaa !8
'i64*8B

	full_text


i64* %35
ígetelementptr8B
}
	full_textp
n
l%37 = getelementptr inbounds [512 x %struct.dcomplex], [512 x %struct.dcomplex]* @cffts3.ty1, i64 0, i64 %22
%i648B

	full_text
	
i64 %22
Kbitcast8B>
<
	full_text/
-
+%38 = bitcast %struct.dcomplex* %37 to i64*
-struct*8B

	full_text

struct* %37
Istore8B>
<
	full_text/
-
+store i64 %36, i64* %38, align 16, !tbaa !8
%i648B

	full_text
	
i64 %36
'i64*8B

	full_text


i64* %38
ygetelementptr8Bf
d
	full_textW
U
S%39 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %0, i64 %33, i32 1
%i648B

	full_text
	
i64 %33
Abitcast8B4
2
	full_text%
#
!%40 = bitcast double* %39 to i64*
-double*8B

	full_text

double* %39
Iload8B?
=
	full_text0
.
,%41 = load i64, i64* %40, align 8, !tbaa !13
'i64*8B

	full_text


i64* %40
õgetelementptr8Bá
Ñ
	full_textw
u
s%42 = getelementptr inbounds [512 x %struct.dcomplex], [512 x %struct.dcomplex]* @cffts3.ty1, i64 0, i64 %22, i32 1
%i648B

	full_text
	
i64 %22
Abitcast8B4
2
	full_text%
#
!%43 = bitcast double* %42 to i64*
-double*8B

	full_text

double* %42
Istore8B>
<
	full_text/
-
+store i64 %41, i64* %43, align 8, !tbaa !13
%i648B

	full_text
	
i64 %41
'i64*8B

	full_text


i64* %43
5add8B,
*
	full_text

%44 = add nsw i64 %22, 64
%i648B

	full_text
	
i64 %22
'br8B

	full_text

br label %45
Dphi8B;
9
	full_text,
*
(%46 = phi i64 [ %22, %19 ], [ %44, %29 ]
%i648B

	full_text
	
i64 %22
%i648B

	full_text
	
i64 %44
5icmp8B+
)
	full_text

%47 = icmp eq i64 %26, 0
%i648B

	full_text
	
i64 %26
:br8B2
0
	full_text#
!
br i1 %47, label %49, label %48
#i18B

	full_text


i1 %47
'br8B

	full_text

br label %84
Rphi8BI
G
	full_text:
8
6%50 = phi i1 [ false, %8 ], [ %18, %84 ], [ %18, %45 ]
#i18B

	full_text


i1 %18
#i18B

	full_text


i1 %18
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
‹call8B—
Œ
	full_text¿
Ω
∫tail call void @cfftz(i32 %3, i32 %7, i32 %6, %struct.dcomplex* %2, %struct.dcomplex* getelementptr inbounds ([512 x %struct.dcomplex], [512 x %struct.dcomplex]* @cffts3.ty1, i64 0, i64 0), %struct.dcomplex* getelementptr inbounds ([512 x %struct.dcomplex], [512 x %struct.dcomplex]* @cffts3.ty2, i64 0, i64 0)) #6
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
;br8B3
1
	full_text$
"
 br i1 %50, label %51, label %117
#i18B

	full_text


i1 %50
1mul8B(
&
	full_text

%52 = mul i32 %15, %5
%i328B

	full_text
	
i32 %15
1shl8B(
&
	full_text

%53 = shl i64 %13, 32
%i648B

	full_text
	
i64 %13
9ashr8B/
-
	full_text 

%54 = ashr exact i64 %53, 32
%i648B

	full_text
	
i64 %53
5sext8B+
)
	full_text

%55 = sext i32 %6 to i64
5add8B,
*
	full_text

%56 = add nsw i64 %54, 64
%i648B

	full_text
	
i64 %54
8icmp8B.
,
	full_text

%57 = icmp sgt i64 %56, %55
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %55
Dselect8B8
6
	full_text)
'
%%58 = select i1 %57, i64 %56, i64 %55
#i18B

	full_text


i1 %57
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %55
5add8B,
*
	full_text

%59 = add nsw i64 %58, -1
%i648B

	full_text
	
i64 %58
6sub8B-
+
	full_text

%60 = sub nsw i64 %59, %54
%i648B

	full_text
	
i64 %59
%i648B

	full_text
	
i64 %54
2lshr8B(
&
	full_text

%61 = lshr i64 %60, 6
%i648B

	full_text
	
i64 %60
0and8B'
%
	full_text

%62 = and i64 %61, 1
%i648B

	full_text
	
i64 %61
5icmp8B+
)
	full_text

%63 = icmp eq i64 %62, 0
%i648B

	full_text
	
i64 %62
:br8B2
0
	full_text#
!
br i1 %63, label %64, label %80
#i18B

	full_text


i1 %63
8trunc8B-
+
	full_text

%65 = trunc i64 %54 to i32
%i648B

	full_text
	
i64 %54
2mul8B)
'
	full_text

%66 = mul i32 %52, %65
%i328B

	full_text
	
i32 %52
%i328B

	full_text
	
i32 %65
6add8B-
+
	full_text

%67 = add nsw i32 %66, %17
%i328B

	full_text
	
i32 %66
%i328B

	full_text
	
i32 %17
ígetelementptr8B
}
	full_textp
n
l%68 = getelementptr inbounds [512 x %struct.dcomplex], [512 x %struct.dcomplex]* @cffts3.ty1, i64 0, i64 %54
%i648B

	full_text
	
i64 %54
Kbitcast8B>
<
	full_text/
-
+%69 = bitcast %struct.dcomplex* %68 to i64*
-struct*8B

	full_text

struct* %68
Iload8B?
=
	full_text0
.
,%70 = load i64, i64* %69, align 16, !tbaa !8
'i64*8B

	full_text


i64* %69
6sext8B,
*
	full_text

%71 = sext i32 %67 to i64
%i328B

	full_text
	
i32 %67
rgetelementptr8B_
]
	full_textP
N
L%72 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %1, i64 %71
%i648B

	full_text
	
i64 %71
Kbitcast8B>
<
	full_text/
-
+%73 = bitcast %struct.dcomplex* %72 to i64*
-struct*8B

	full_text

struct* %72
Hstore8B=
;
	full_text.
,
*store i64 %70, i64* %73, align 8, !tbaa !8
%i648B

	full_text
	
i64 %70
'i64*8B

	full_text


i64* %73
õgetelementptr8Bá
Ñ
	full_textw
u
s%74 = getelementptr inbounds [512 x %struct.dcomplex], [512 x %struct.dcomplex]* @cffts3.ty1, i64 0, i64 %54, i32 1
%i648B

	full_text
	
i64 %54
Abitcast8B4
2
	full_text%
#
!%75 = bitcast double* %74 to i64*
-double*8B

	full_text

double* %74
Iload8B?
=
	full_text0
.
,%76 = load i64, i64* %75, align 8, !tbaa !13
'i64*8B

	full_text


i64* %75
ygetelementptr8Bf
d
	full_textW
U
S%77 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %1, i64 %71, i32 1
%i648B

	full_text
	
i64 %71
Abitcast8B4
2
	full_text%
#
!%78 = bitcast double* %77 to i64*
-double*8B

	full_text

double* %77
Istore8B>
<
	full_text/
-
+store i64 %76, i64* %78, align 8, !tbaa !13
%i648B

	full_text
	
i64 %76
'i64*8B

	full_text


i64* %78
5add8B,
*
	full_text

%79 = add nsw i64 %54, 64
%i648B

	full_text
	
i64 %54
'br8B

	full_text

br label %80
Dphi8B;
9
	full_text,
*
(%81 = phi i64 [ %54, %51 ], [ %79, %64 ]
%i648B

	full_text
	
i64 %54
%i648B

	full_text
	
i64 %79
5icmp8B+
)
	full_text

%82 = icmp eq i64 %61, 0
%i648B

	full_text
	
i64 %61
;br8B3
1
	full_text$
"
 br i1 %82, label %117, label %83
#i18B

	full_text


i1 %82
(br8	B 

	full_text

br label %118
Ephi8
B<
:
	full_text-
+
)%85 = phi i64 [ %46, %48 ], [ %115, %84 ]
%i648
B

	full_text
	
i64 %46
&i648
B

	full_text


i64 %115
8trunc8
B-
+
	full_text

%86 = trunc i64 %85 to i32
%i648
B

	full_text
	
i64 %85
2mul8
B)
'
	full_text

%87 = mul i32 %20, %86
%i328
B

	full_text
	
i32 %20
%i328
B

	full_text
	
i32 %86
6add8
B-
+
	full_text

%88 = add nsw i32 %87, %17
%i328
B

	full_text
	
i32 %87
%i328
B

	full_text
	
i32 %17
6sext8
B,
*
	full_text

%89 = sext i32 %88 to i64
%i328
B

	full_text
	
i32 %88
rgetelementptr8
B_
]
	full_textP
N
L%90 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %0, i64 %89
%i648
B

	full_text
	
i64 %89
Kbitcast8
B>
<
	full_text/
-
+%91 = bitcast %struct.dcomplex* %90 to i64*
-struct*8
B

	full_text

struct* %90
Hload8
B>
<
	full_text/
-
+%92 = load i64, i64* %91, align 8, !tbaa !8
'i64*8
B

	full_text


i64* %91
ígetelementptr8
B
}
	full_textp
n
l%93 = getelementptr inbounds [512 x %struct.dcomplex], [512 x %struct.dcomplex]* @cffts3.ty1, i64 0, i64 %85
%i648
B

	full_text
	
i64 %85
Kbitcast8
B>
<
	full_text/
-
+%94 = bitcast %struct.dcomplex* %93 to i64*
-struct*8
B

	full_text

struct* %93
Istore8
B>
<
	full_text/
-
+store i64 %92, i64* %94, align 16, !tbaa !8
%i648
B

	full_text
	
i64 %92
'i64*8
B

	full_text


i64* %94
ygetelementptr8
Bf
d
	full_textW
U
S%95 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %0, i64 %89, i32 1
%i648
B

	full_text
	
i64 %89
Abitcast8
B4
2
	full_text%
#
!%96 = bitcast double* %95 to i64*
-double*8
B

	full_text

double* %95
Iload8
B?
=
	full_text0
.
,%97 = load i64, i64* %96, align 8, !tbaa !13
'i64*8
B

	full_text


i64* %96
õgetelementptr8
Bá
Ñ
	full_textw
u
s%98 = getelementptr inbounds [512 x %struct.dcomplex], [512 x %struct.dcomplex]* @cffts3.ty1, i64 0, i64 %85, i32 1
%i648
B

	full_text
	
i64 %85
Abitcast8
B4
2
	full_text%
#
!%99 = bitcast double* %98 to i64*
-double*8
B

	full_text

double* %98
Istore8
B>
<
	full_text/
-
+store i64 %97, i64* %99, align 8, !tbaa !13
%i648
B

	full_text
	
i64 %97
'i64*8
B

	full_text


i64* %99
6add8
B-
+
	full_text

%100 = add nsw i64 %85, 64
%i648
B

	full_text
	
i64 %85
:trunc8
B/
-
	full_text 

%101 = trunc i64 %100 to i32
&i648
B

	full_text


i64 %100
4mul8
B+
)
	full_text

%102 = mul i32 %20, %101
%i328
B

	full_text
	
i32 %20
&i328
B

	full_text


i32 %101
8add8
B/
-
	full_text 

%103 = add nsw i32 %102, %17
&i328
B

	full_text


i32 %102
%i328
B

	full_text
	
i32 %17
8sext8
B.
,
	full_text

%104 = sext i32 %103 to i64
&i328
B

	full_text


i32 %103
tgetelementptr8
Ba
_
	full_textR
P
N%105 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %0, i64 %104
&i648
B

	full_text


i64 %104
Mbitcast8
B@
>
	full_text1
/
-%106 = bitcast %struct.dcomplex* %105 to i64*
.struct*8
B

	full_text

struct* %105
Jload8
B@
>
	full_text1
/
-%107 = load i64, i64* %106, align 8, !tbaa !8
(i64*8
B

	full_text

	i64* %106
ïgetelementptr8
BÅ

	full_textr
p
n%108 = getelementptr inbounds [512 x %struct.dcomplex], [512 x %struct.dcomplex]* @cffts3.ty1, i64 0, i64 %100
&i648
B

	full_text


i64 %100
Mbitcast8
B@
>
	full_text1
/
-%109 = bitcast %struct.dcomplex* %108 to i64*
.struct*8
B

	full_text

struct* %108
Kstore8
B@
>
	full_text1
/
-store i64 %107, i64* %109, align 16, !tbaa !8
&i648
B

	full_text


i64 %107
(i64*8
B

	full_text

	i64* %109
{getelementptr8
Bh
f
	full_textY
W
U%110 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %0, i64 %104, i32 1
&i648
B

	full_text


i64 %104
Cbitcast8
B6
4
	full_text'
%
#%111 = bitcast double* %110 to i64*
.double*8
B

	full_text

double* %110
Kload8
BA
?
	full_text2
0
.%112 = load i64, i64* %111, align 8, !tbaa !13
(i64*8
B

	full_text

	i64* %111
ùgetelementptr8
Bâ
Ü
	full_texty
w
u%113 = getelementptr inbounds [512 x %struct.dcomplex], [512 x %struct.dcomplex]* @cffts3.ty1, i64 0, i64 %100, i32 1
&i648
B

	full_text


i64 %100
Cbitcast8
B6
4
	full_text'
%
#%114 = bitcast double* %113 to i64*
.double*8
B

	full_text

double* %113
Kstore8
B@
>
	full_text1
/
-store i64 %112, i64* %114, align 8, !tbaa !13
&i648
B

	full_text


i64 %112
(i64*8
B

	full_text

	i64* %114
7add8
B.
,
	full_text

%115 = add nsw i64 %85, 128
%i648
B

	full_text
	
i64 %85
:icmp8
B0
.
	full_text!

%116 = icmp slt i64 %115, %23
&i648
B

	full_text


i64 %115
%i648
B

	full_text
	
i64 %23
;br8
B3
1
	full_text$
"
 br i1 %116, label %84, label %49
$i18
B

	full_text
	
i1 %116
$ret8B

	full_text


ret void
Gphi8B>
<
	full_text/
-
+%119 = phi i64 [ %81, %83 ], [ %149, %118 ]
%i648B

	full_text
	
i64 %81
&i648B

	full_text


i64 %149
:trunc8B/
-
	full_text 

%120 = trunc i64 %119 to i32
&i648B

	full_text


i64 %119
4mul8B+
)
	full_text

%121 = mul i32 %52, %120
%i328B

	full_text
	
i32 %52
&i328B

	full_text


i32 %120
8add8B/
-
	full_text 

%122 = add nsw i32 %121, %17
&i328B

	full_text


i32 %121
%i328B

	full_text
	
i32 %17
ïgetelementptr8BÅ

	full_textr
p
n%123 = getelementptr inbounds [512 x %struct.dcomplex], [512 x %struct.dcomplex]* @cffts3.ty1, i64 0, i64 %119
&i648B

	full_text


i64 %119
Mbitcast8B@
>
	full_text1
/
-%124 = bitcast %struct.dcomplex* %123 to i64*
.struct*8B

	full_text

struct* %123
Kload8BA
?
	full_text2
0
.%125 = load i64, i64* %124, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %124
8sext8B.
,
	full_text

%126 = sext i32 %122 to i64
&i328B

	full_text


i32 %122
tgetelementptr8Ba
_
	full_textR
P
N%127 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %1, i64 %126
&i648B

	full_text


i64 %126
Mbitcast8B@
>
	full_text1
/
-%128 = bitcast %struct.dcomplex* %127 to i64*
.struct*8B

	full_text

struct* %127
Jstore8B?
=
	full_text0
.
,store i64 %125, i64* %128, align 8, !tbaa !8
&i648B

	full_text


i64 %125
(i64*8B

	full_text

	i64* %128
ùgetelementptr8Bâ
Ü
	full_texty
w
u%129 = getelementptr inbounds [512 x %struct.dcomplex], [512 x %struct.dcomplex]* @cffts3.ty1, i64 0, i64 %119, i32 1
&i648B

	full_text


i64 %119
Cbitcast8B6
4
	full_text'
%
#%130 = bitcast double* %129 to i64*
.double*8B

	full_text

double* %129
Kload8BA
?
	full_text2
0
.%131 = load i64, i64* %130, align 8, !tbaa !13
(i64*8B

	full_text

	i64* %130
{getelementptr8Bh
f
	full_textY
W
U%132 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %1, i64 %126, i32 1
&i648B

	full_text


i64 %126
Cbitcast8B6
4
	full_text'
%
#%133 = bitcast double* %132 to i64*
.double*8B

	full_text

double* %132
Kstore8B@
>
	full_text1
/
-store i64 %131, i64* %133, align 8, !tbaa !13
&i648B

	full_text


i64 %131
(i64*8B

	full_text

	i64* %133
7add8B.
,
	full_text

%134 = add nsw i64 %119, 64
&i648B

	full_text


i64 %119
:trunc8B/
-
	full_text 

%135 = trunc i64 %134 to i32
&i648B

	full_text


i64 %134
4mul8B+
)
	full_text

%136 = mul i32 %52, %135
%i328B

	full_text
	
i32 %52
&i328B

	full_text


i32 %135
8add8B/
-
	full_text 

%137 = add nsw i32 %136, %17
&i328B

	full_text


i32 %136
%i328B

	full_text
	
i32 %17
ïgetelementptr8BÅ

	full_textr
p
n%138 = getelementptr inbounds [512 x %struct.dcomplex], [512 x %struct.dcomplex]* @cffts3.ty1, i64 0, i64 %134
&i648B

	full_text


i64 %134
Mbitcast8B@
>
	full_text1
/
-%139 = bitcast %struct.dcomplex* %138 to i64*
.struct*8B

	full_text

struct* %138
Kload8BA
?
	full_text2
0
.%140 = load i64, i64* %139, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %139
8sext8B.
,
	full_text

%141 = sext i32 %137 to i64
&i328B

	full_text


i32 %137
tgetelementptr8Ba
_
	full_textR
P
N%142 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %1, i64 %141
&i648B

	full_text


i64 %141
Mbitcast8B@
>
	full_text1
/
-%143 = bitcast %struct.dcomplex* %142 to i64*
.struct*8B

	full_text

struct* %142
Jstore8B?
=
	full_text0
.
,store i64 %140, i64* %143, align 8, !tbaa !8
&i648B

	full_text


i64 %140
(i64*8B

	full_text

	i64* %143
ùgetelementptr8Bâ
Ü
	full_texty
w
u%144 = getelementptr inbounds [512 x %struct.dcomplex], [512 x %struct.dcomplex]* @cffts3.ty1, i64 0, i64 %134, i32 1
&i648B

	full_text


i64 %134
Cbitcast8B6
4
	full_text'
%
#%145 = bitcast double* %144 to i64*
.double*8B

	full_text

double* %144
Kload8BA
?
	full_text2
0
.%146 = load i64, i64* %145, align 8, !tbaa !13
(i64*8B

	full_text

	i64* %145
{getelementptr8Bh
f
	full_textY
W
U%147 = getelementptr inbounds %struct.dcomplex, %struct.dcomplex* %1, i64 %141, i32 1
&i648B

	full_text


i64 %141
Cbitcast8B6
4
	full_text'
%
#%148 = bitcast double* %147 to i64*
.double*8B

	full_text

double* %147
Kstore8B@
>
	full_text1
/
-store i64 %146, i64* %148, align 8, !tbaa !13
&i648B

	full_text


i64 %146
(i64*8B

	full_text

	i64* %148
8add8B/
-
	full_text 

%149 = add nsw i64 %119, 128
&i648B

	full_text


i64 %119
:icmp8B0
.
	full_text!

%150 = icmp slt i64 %149, %55
&i648B

	full_text


i64 %149
%i648B

	full_text
	
i64 %55
=br8B5
3
	full_text&
$
"br i1 %150, label %118, label %117
$i18B

	full_text
	
i1 %150
$i328B

	full_text


i32 %6
$i328B

	full_text


i32 %7
6struct*8B'
%
	full_text

%struct.dcomplex* %0
6struct*8B'
%
	full_text

%struct.dcomplex* %1
$i328B

	full_text


i32 %5
$i328B

	full_text


i32 %3
6struct*8B'
%
	full_text

%struct.dcomplex* %2
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
$i648B

	full_text


i64 -1
z[512 x %struct.dcomplex]*8BY
W
	full_textJ
H
F@cffts3.ty1 = internal global [512 x %struct.dcomplex] undef, align 16
%i18B

	full_text


i1 false
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 6
$i648B

	full_text


i64 32
$i648B

	full_text


i64 64
ústruct*8Bå
â
	full_text|
z
x%struct.dcomplex* getelementptr inbounds ([512 x %struct.dcomplex], [512 x %struct.dcomplex]* @cffts3.ty1, i64 0, i64 0)
%i648B

	full_text
	
i64 128
#i328B

	full_text	

i32 1
#i328B

	full_text	

i32 0
ústruct*8Bå
â
	full_text|
z
x%struct.dcomplex* getelementptr inbounds ([512 x %struct.dcomplex], [512 x %struct.dcomplex]* @cffts3.ty2, i64 0, i64 0)       	  

                       !" !! #$ ## %& %% '( '* )) +, +- ++ ./ .0 .. 12 11 34 33 56 55 78 77 9: 99 ;< ;; => =? == @A @@ BC BB DE DD FG FF HI HH JK JL JJ MN MM OQ PR PP ST SS UV UY XZ XX [[ \\ ]] ^_ ^a `` bc bb de dd ff gh gg ij ik ii lm ln lo ll pq pp rs rt rr uv uu wx ww yz yy {| {~ }} Ä 	Å  ÇÉ Ç
Ñ ÇÇ Ö
Ü ÖÖ áà áá âä ââ ãå ãã ç
é çç èê èè ëí ë
ì ëë î
ï îî ñó ññ òô òò ö
õ öö úù úú ûü û
† ûû °¢ °° £• §
¶ §§ ß® ßß ©™ ©≠ ¨
Æ ¨¨ Ø∞ ØØ ±≤ ±
≥ ±± ¥µ ¥
∂ ¥¥ ∑∏ ∑∑ π
∫ ππ ªº ªª Ωæ ΩΩ ø
¿ øø ¡¬ ¡¡ √ƒ √
≈ √√ ∆
« ∆∆ »… »»  À    Ã
Õ ÃÃ Œœ ŒŒ –— –
“ –– ”‘ ”” ’÷ ’’ ◊ÿ ◊
Ÿ ◊◊ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›› ﬂ
‡ ﬂﬂ ·‚ ·· „‰ „„ Â
Ê ÂÂ ÁË ÁÁ ÈÍ È
Î ÈÈ Ï
Ì ÏÏ ÓÔ ÓÓ Ò  Ú
Û ÚÚ Ùı ÙÙ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘˘ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛Ç Å
É ÅÅ ÑÖ ÑÑ Üá Ü
à ÜÜ âä â
ã ââ å
ç åå éè éé êë êê íì íí î
ï îî ñó ññ òô ò
ö òò õ
ú õõ ùû ùù ü† üü °
¢ °° £§ ££ •¶ •
ß •• ®© ®® ™´ ™™ ¨≠ ¨
Æ ¨¨ Ø∞ Ø
± ØØ ≤
≥ ≤≤ ¥µ ¥¥ ∂∑ ∂∂ ∏π ∏∏ ∫
ª ∫∫ ºΩ ºº æø æ
¿ ææ ¡
¬ ¡¡ √ƒ √√ ≈∆ ≈≈ «
» «« …  …… ÀÃ À
Õ ÀÀ Œœ ŒŒ –— –
“ –– ”‘ ”	’ ’ 	’ \’ f	÷ \◊ 3◊ @◊ π◊ ∆◊ ﬂ◊ Ïÿ çÿ öÿ îÿ °ÿ ∫ÿ «	Ÿ 	Ÿ `⁄ \	€ \‹ 
   	
      
        "! $# &% ( * ,) -+ / 0. 21 43 65 8 :9 <7 >; ?1 A@ CB E GF ID KH L N QM R! TS V Y ZX _
 a cb ed hg jf ki mg nf ol qp sd tr vu xw zy |d ~` Ä} Å É Ñd ÜÖ àá äÇ åã éç êâ íè ìd ïî óñ ôã õö ùò üú †d ¢d •° ¶u ®ß ™P ≠˘ Æ¨ ∞ ≤Ø ≥± µ ∂¥ ∏∑ ∫π ºª æ¨ ¿ø ¬Ω ƒ¡ ≈∑ «∆ …» À¨ ÕÃ œ  —Œ “¨ ‘” ÷ ÿ’ Ÿ◊ € ‹⁄ ﬁ› ‡ﬂ ‚· ‰” ÊÂ Ë„ ÍÁ Î› ÌÏ ÔÓ Ò” ÛÚ ı ˜Ù ¯¨ ˙˘ ¸ ˝˚ ˇ§ ÇŒ ÉÅ Ö` áÑ àÜ ä ãÅ çå èé ëâ ìí ïî óê ôñ öÅ úõ ûù †í ¢° §ü ¶£ ßÅ ©® ´` ≠™ Æ¨ ∞ ±® ≥≤ µ¥ ∑Ø π∏ ª∫ Ω∂ øº ¿® ¬¡ ƒ√ ∆∏ »«  ≈ Ã… ÕÅ œŒ —f “– ‘  X' )' P^ `^ ÄO PU XU W{ }{ §W ¨£ §© Ä© ´˛ ¨˛ X´ Å” Å” Ä Ä ﬁﬁ ﬂﬂ ‡‡ ›› ··\ ·· \ ﬁﬁ  ›› [ ‡‡ [ ﬂﬂ ] ‡‡ ]	‚ 	‚ p„ 9„ F„ Ö„ î„ ø„ Ã„ Â„ Ú„ å„ õ„ ≤„ ¡‰ X	Â #	Â w	Ê %	Ê 9	Ê F	Ê S	Ê y
Ê Ö
Ê î
Ê ß
Ê ø
Ê Ã
Ê Â
Ê Ú
Ê å
Ê õ
Ê ≤
Ê ¡	Á !	Á u	Ë 	Ë 	Ë b	Ë d	È M	È g
È °
È ”
È ®	Í \
Î ˘
Î ŒÏ 	Ï 
	Ï @	Ï FÏ [Ï ]
Ï î
Ï ö
Ï ∆
Ï Ã
Ï Ï
Ï Ú
Ï õ
Ï °
Ï ¡
Ï «Ì Ì 	Ó \"
cffts3"
_Z13get_global_idj"
_Z12get_group_idj"
_Z12get_local_idj"
_Z7barrierj"
cfftz*â
npb-FT-cffts3.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Å
 
transfer_bytes_log1p
˙'®A

wgsize_log1p
˙'®A

transfer_bytes	
ê‰†Å

devmap_label


wgsize
@