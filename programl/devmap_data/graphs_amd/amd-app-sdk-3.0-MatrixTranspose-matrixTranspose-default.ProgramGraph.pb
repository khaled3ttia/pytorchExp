

[external]
McallBE
C
	full_text6
4
2%4 = tail call i64 @_Z15get_global_sizej(i32 0) #3
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
JcallBB
@
	full_text3
1
/%6 = tail call i64 @_Z12get_group_idj(i32 0) #3
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
JcallBB
@
	full_text3
1
/%8 = tail call i64 @_Z12get_group_idj(i32 1) #3
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
McallBE
C
	full_text6
4
2%10 = tail call i64 @_Z14get_num_groupsj(i32 0) #3
6truncB-
+
	full_text

%11 = trunc i64 %10 to i32
#i64B

	full_text
	
i64 %10
.addB'
%
	full_text

%12 = add i32 %9, %7
"i32B

	full_text


i32 %9
"i32B

	full_text


i32 %7
2uremB*
(
	full_text

%13 = urem i32 %12, %11
#i32B

	full_text
	
i32 %12
#i32B

	full_text
	
i32 %11
KcallBC
A
	full_text4
2
0%14 = tail call i64 @_Z12get_local_idj(i32 0) #3
6truncB-
+
	full_text

%15 = trunc i64 %14 to i32
#i64B

	full_text
	
i64 %14
KcallBC
A
	full_text4
2
0%16 = tail call i64 @_Z12get_local_idj(i32 1) #3
6truncB-
+
	full_text

%17 = trunc i64 %16 to i32
#i64B

	full_text
	
i64 %16
McallBE
C
	full_text6
4
2%18 = tail call i64 @_Z14get_local_sizej(i32 0) #3
6truncB-
+
	full_text

%19 = trunc i64 %18 to i32
#i64B

	full_text
	
i64 %18
0mulB)
'
	full_text

%20 = mul i32 %13, %19
#i32B

	full_text
	
i32 %13
#i32B

	full_text
	
i32 %19
0addB)
'
	full_text

%21 = add i32 %20, %15
#i32B

	full_text
	
i32 %20
#i32B

	full_text
	
i32 %15
/mulB(
&
	full_text

%22 = mul i32 %19, %7
#i32B

	full_text
	
i32 %19
"i32B

	full_text


i32 %7
0addB)
'
	full_text

%23 = add i32 %22, %17
#i32B

	full_text
	
i32 %22
#i32B

	full_text
	
i32 %17
-shlB&
$
	full_text

%24 = shl i32 %5, 2
"i32B

	full_text


i32 %5
0mulB)
'
	full_text

%25 = mul i32 %24, %23
#i32B

	full_text
	
i32 %24
#i32B

	full_text
	
i32 %23
0addB)
'
	full_text

%26 = add i32 %21, %25
#i32B

	full_text
	
i32 %21
#i32B

	full_text
	
i32 %25
.shlB'
%
	full_text

%27 = shl i32 %17, 2
#i32B

	full_text
	
i32 %17
0mulB)
'
	full_text

%28 = mul i32 %27, %19
#i32B

	full_text
	
i32 %27
#i32B

	full_text
	
i32 %19
0addB)
'
	full_text

%29 = add i32 %28, %15
#i32B

	full_text
	
i32 %28
#i32B

	full_text
	
i32 %15
4sextB,
*
	full_text

%30 = sext i32 %26 to i64
#i32B

	full_text
	
i32 %26
fgetelementptrBU
S
	full_textF
D
B%31 = getelementptr inbounds <4 x float>, <4 x float>* %1, i64 %30
#i64B

	full_text
	
i64 %30
WloadBO
M
	full_text@
>
<%32 = load <4 x float>, <4 x float>* %31, align 16, !tbaa !9
5<4 x float>*B#
!
	full_text

<4 x float>* %31
4sextB,
*
	full_text

%33 = sext i32 %29 to i64
#i32B

	full_text
	
i32 %29
fgetelementptrBU
S
	full_textF
D
B%34 = getelementptr inbounds <4 x float>, <4 x float>* %2, i64 %33
#i64B

	full_text
	
i64 %33
WstoreBN
L
	full_text?
=
;store <4 x float> %32, <4 x float>* %34, align 16, !tbaa !9
3<4 x float>B"
 
	full_text

<4 x float> %32
5<4 x float>*B#
!
	full_text

<4 x float>* %34
/addB(
&
	full_text

%35 = add i32 %26, %5
#i32B

	full_text
	
i32 %26
"i32B

	full_text


i32 %5
4zextB,
*
	full_text

%36 = zext i32 %35 to i64
#i32B

	full_text
	
i32 %35
fgetelementptrBU
S
	full_textF
D
B%37 = getelementptr inbounds <4 x float>, <4 x float>* %1, i64 %36
#i64B

	full_text
	
i64 %36
WloadBO
M
	full_text@
>
<%38 = load <4 x float>, <4 x float>* %37, align 16, !tbaa !9
5<4 x float>*B#
!
	full_text

<4 x float>* %37
0addB)
'
	full_text

%39 = add i32 %29, %19
#i32B

	full_text
	
i32 %29
#i32B

	full_text
	
i32 %19
4zextB,
*
	full_text

%40 = zext i32 %39 to i64
#i32B

	full_text
	
i32 %39
fgetelementptrBU
S
	full_textF
D
B%41 = getelementptr inbounds <4 x float>, <4 x float>* %2, i64 %40
#i64B

	full_text
	
i64 %40
WstoreBN
L
	full_text?
=
;store <4 x float> %38, <4 x float>* %41, align 16, !tbaa !9
3<4 x float>B"
 
	full_text

<4 x float> %38
5<4 x float>*B#
!
	full_text

<4 x float>* %41
-shlB&
$
	full_text

%42 = shl i32 %5, 1
"i32B

	full_text


i32 %5
0addB)
'
	full_text

%43 = add i32 %26, %42
#i32B

	full_text
	
i32 %26
#i32B

	full_text
	
i32 %42
4zextB,
*
	full_text

%44 = zext i32 %43 to i64
#i32B

	full_text
	
i32 %43
fgetelementptrBU
S
	full_textF
D
B%45 = getelementptr inbounds <4 x float>, <4 x float>* %1, i64 %44
#i64B

	full_text
	
i64 %44
WloadBO
M
	full_text@
>
<%46 = load <4 x float>, <4 x float>* %45, align 16, !tbaa !9
5<4 x float>*B#
!
	full_text

<4 x float>* %45
.shlB'
%
	full_text

%47 = shl i32 %19, 1
#i32B

	full_text
	
i32 %19
0addB)
'
	full_text

%48 = add i32 %29, %47
#i32B

	full_text
	
i32 %29
#i32B

	full_text
	
i32 %47
4zextB,
*
	full_text

%49 = zext i32 %48 to i64
#i32B

	full_text
	
i32 %48
fgetelementptrBU
S
	full_textF
D
B%50 = getelementptr inbounds <4 x float>, <4 x float>* %2, i64 %49
#i64B

	full_text
	
i64 %49
WstoreBN
L
	full_text?
=
;store <4 x float> %46, <4 x float>* %50, align 16, !tbaa !9
3<4 x float>B"
 
	full_text

<4 x float> %46
5<4 x float>*B#
!
	full_text

<4 x float>* %50
-mulB&
$
	full_text

%51 = mul i32 %5, 3
"i32B

	full_text


i32 %5
0addB)
'
	full_text

%52 = add i32 %26, %51
#i32B

	full_text
	
i32 %26
#i32B

	full_text
	
i32 %51
4zextB,
*
	full_text

%53 = zext i32 %52 to i64
#i32B

	full_text
	
i32 %52
fgetelementptrBU
S
	full_textF
D
B%54 = getelementptr inbounds <4 x float>, <4 x float>* %1, i64 %53
#i64B

	full_text
	
i64 %53
WloadBO
M
	full_text@
>
<%55 = load <4 x float>, <4 x float>* %54, align 16, !tbaa !9
5<4 x float>*B#
!
	full_text

<4 x float>* %54
.mulB'
%
	full_text

%56 = mul i32 %19, 3
#i32B

	full_text
	
i32 %19
0addB)
'
	full_text

%57 = add i32 %29, %56
#i32B

	full_text
	
i32 %29
#i32B

	full_text
	
i32 %56
4zextB,
*
	full_text

%58 = zext i32 %57 to i64
#i32B

	full_text
	
i32 %57
fgetelementptrBU
S
	full_textF
D
B%59 = getelementptr inbounds <4 x float>, <4 x float>* %2, i64 %58
#i64B

	full_text
	
i64 %58
WstoreBN
L
	full_text?
=
;store <4 x float> %55, <4 x float>* %59, align 16, !tbaa !9
3<4 x float>B"
 
	full_text

<4 x float> %55
5<4 x float>*B#
!
	full_text

<4 x float>* %59
@callB8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
0addB)
'
	full_text

%60 = add i32 %22, %15
#i32B

	full_text
	
i32 %22
#i32B

	full_text
	
i32 %15
0addB)
'
	full_text

%61 = add i32 %20, %17
#i32B

	full_text
	
i32 %20
#i32B

	full_text
	
i32 %17
0mulB)
'
	full_text

%62 = mul i32 %24, %61
#i32B

	full_text
	
i32 %24
#i32B

	full_text
	
i32 %61
0addB)
'
	full_text

%63 = add i32 %60, %62
#i32B

	full_text
	
i32 %60
#i32B

	full_text
	
i32 %62
.shlB'
%
	full_text

%64 = shl i32 %15, 2
#i32B

	full_text
	
i32 %15
0mulB)
'
	full_text

%65 = mul i32 %64, %19
#i32B

	full_text
	
i32 %64
#i32B

	full_text
	
i32 %19
0addB)
'
	full_text

%66 = add i32 %65, %17
#i32B

	full_text
	
i32 %65
#i32B

	full_text
	
i32 %17
4sextB,
*
	full_text

%67 = sext i32 %66 to i64
#i32B

	full_text
	
i32 %66
fgetelementptrBU
S
	full_textF
D
B%68 = getelementptr inbounds <4 x float>, <4 x float>* %2, i64 %67
#i64B

	full_text
	
i64 %67
WloadBO
M
	full_text@
>
<%69 = load <4 x float>, <4 x float>* %68, align 16, !tbaa !9
5<4 x float>*B#
!
	full_text

<4 x float>* %68
0addB)
'
	full_text

%70 = add i32 %66, %19
#i32B

	full_text
	
i32 %66
#i32B

	full_text
	
i32 %19
4zextB,
*
	full_text

%71 = zext i32 %70 to i64
#i32B

	full_text
	
i32 %70
fgetelementptrBU
S
	full_textF
D
B%72 = getelementptr inbounds <4 x float>, <4 x float>* %2, i64 %71
#i64B

	full_text
	
i64 %71
WloadBO
M
	full_text@
>
<%73 = load <4 x float>, <4 x float>* %72, align 16, !tbaa !9
5<4 x float>*B#
!
	full_text

<4 x float>* %72
0addB)
'
	full_text

%74 = add i32 %66, %47
#i32B

	full_text
	
i32 %66
#i32B

	full_text
	
i32 %47
4zextB,
*
	full_text

%75 = zext i32 %74 to i64
#i32B

	full_text
	
i32 %74
fgetelementptrBU
S
	full_textF
D
B%76 = getelementptr inbounds <4 x float>, <4 x float>* %2, i64 %75
#i64B

	full_text
	
i64 %75
WloadBO
M
	full_text@
>
<%77 = load <4 x float>, <4 x float>* %76, align 16, !tbaa !9
5<4 x float>*B#
!
	full_text

<4 x float>* %76
0addB)
'
	full_text

%78 = add i32 %66, %56
#i32B

	full_text
	
i32 %66
#i32B

	full_text
	
i32 %56
4zextB,
*
	full_text

%79 = zext i32 %78 to i64
#i32B

	full_text
	
i32 %78
fgetelementptrBU
S
	full_textF
D
B%80 = getelementptr inbounds <4 x float>, <4 x float>* %2, i64 %79
#i64B

	full_text
	
i64 %79
WloadBO
M
	full_text@
>
<%81 = load <4 x float>, <4 x float>* %80, align 16, !tbaa !9
5<4 x float>*B#
!
	full_text

<4 x float>* %80
ˆshufflevectorBw
u
	full_texth
f
d%82 = shufflevector <4 x float> %69, <4 x float> %73, <4 x i32> <i32 0, i32 4, i32 undef, i32 undef>
3<4 x float>B"
 
	full_text

<4 x float> %69
3<4 x float>B"
 
	full_text

<4 x float> %73
„shufflevectorBs
q
	full_textd
b
`%83 = shufflevector <4 x float> %82, <4 x float> %77, <4 x i32> <i32 0, i32 1, i32 4, i32 undef>
3<4 x float>B"
 
	full_text

<4 x float> %82
3<4 x float>B"
 
	full_text

<4 x float> %77
€shufflevectorBo
m
	full_text`
^
\%84 = shufflevector <4 x float> %83, <4 x float> %81, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
3<4 x float>B"
 
	full_text

<4 x float> %83
3<4 x float>B"
 
	full_text

<4 x float> %81
4sextB,
*
	full_text

%85 = sext i32 %63 to i64
#i32B

	full_text
	
i32 %63
fgetelementptrBU
S
	full_textF
D
B%86 = getelementptr inbounds <4 x float>, <4 x float>* %0, i64 %85
#i64B

	full_text
	
i64 %85
WstoreBN
L
	full_text?
=
;store <4 x float> %84, <4 x float>* %86, align 16, !tbaa !9
3<4 x float>B"
 
	full_text

<4 x float> %84
5<4 x float>*B#
!
	full_text

<4 x float>* %86
ˆshufflevectorBw
u
	full_texth
f
d%87 = shufflevector <4 x float> %69, <4 x float> %73, <4 x i32> <i32 1, i32 5, i32 undef, i32 undef>
3<4 x float>B"
 
	full_text

<4 x float> %69
3<4 x float>B"
 
	full_text

<4 x float> %73
„shufflevectorBs
q
	full_textd
b
`%88 = shufflevector <4 x float> %87, <4 x float> %77, <4 x i32> <i32 0, i32 1, i32 5, i32 undef>
3<4 x float>B"
 
	full_text

<4 x float> %87
3<4 x float>B"
 
	full_text

<4 x float> %77
€shufflevectorBo
m
	full_text`
^
\%89 = shufflevector <4 x float> %88, <4 x float> %81, <4 x i32> <i32 0, i32 1, i32 2, i32 5>
3<4 x float>B"
 
	full_text

<4 x float> %88
3<4 x float>B"
 
	full_text

<4 x float> %81
/addB(
&
	full_text

%90 = add i32 %63, %5
#i32B

	full_text
	
i32 %63
"i32B

	full_text


i32 %5
4zextB,
*
	full_text

%91 = zext i32 %90 to i64
#i32B

	full_text
	
i32 %90
fgetelementptrBU
S
	full_textF
D
B%92 = getelementptr inbounds <4 x float>, <4 x float>* %0, i64 %91
#i64B

	full_text
	
i64 %91
WstoreBN
L
	full_text?
=
;store <4 x float> %89, <4 x float>* %92, align 16, !tbaa !9
3<4 x float>B"
 
	full_text

<4 x float> %89
5<4 x float>*B#
!
	full_text

<4 x float>* %92
ˆshufflevectorBw
u
	full_texth
f
d%93 = shufflevector <4 x float> %69, <4 x float> %73, <4 x i32> <i32 2, i32 6, i32 undef, i32 undef>
3<4 x float>B"
 
	full_text

<4 x float> %69
3<4 x float>B"
 
	full_text

<4 x float> %73
„shufflevectorBs
q
	full_textd
b
`%94 = shufflevector <4 x float> %93, <4 x float> %77, <4 x i32> <i32 0, i32 1, i32 6, i32 undef>
3<4 x float>B"
 
	full_text

<4 x float> %93
3<4 x float>B"
 
	full_text

<4 x float> %77
€shufflevectorBo
m
	full_text`
^
\%95 = shufflevector <4 x float> %94, <4 x float> %81, <4 x i32> <i32 0, i32 1, i32 2, i32 6>
3<4 x float>B"
 
	full_text

<4 x float> %94
3<4 x float>B"
 
	full_text

<4 x float> %81
0addB)
'
	full_text

%96 = add i32 %63, %42
#i32B

	full_text
	
i32 %63
#i32B

	full_text
	
i32 %42
4zextB,
*
	full_text

%97 = zext i32 %96 to i64
#i32B

	full_text
	
i32 %96
fgetelementptrBU
S
	full_textF
D
B%98 = getelementptr inbounds <4 x float>, <4 x float>* %0, i64 %97
#i64B

	full_text
	
i64 %97
WstoreBN
L
	full_text?
=
;store <4 x float> %95, <4 x float>* %98, align 16, !tbaa !9
3<4 x float>B"
 
	full_text

<4 x float> %95
5<4 x float>*B#
!
	full_text

<4 x float>* %98
ˆshufflevectorBw
u
	full_texth
f
d%99 = shufflevector <4 x float> %69, <4 x float> %73, <4 x i32> <i32 3, i32 7, i32 undef, i32 undef>
3<4 x float>B"
 
	full_text

<4 x float> %69
3<4 x float>B"
 
	full_text

<4 x float> %73
…shufflevectorBt
r
	full_texte
c
a%100 = shufflevector <4 x float> %99, <4 x float> %77, <4 x i32> <i32 0, i32 1, i32 7, i32 undef>
3<4 x float>B"
 
	full_text

<4 x float> %99
3<4 x float>B"
 
	full_text

<4 x float> %77
‚shufflevectorBq
o
	full_textb
`
^%101 = shufflevector <4 x float> %100, <4 x float> %81, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
4<4 x float>B#
!
	full_text

<4 x float> %100
3<4 x float>B"
 
	full_text

<4 x float> %81
1addB*
(
	full_text

%102 = add i32 %63, %51
#i32B

	full_text
	
i32 %63
#i32B

	full_text
	
i32 %51
6zextB.
,
	full_text

%103 = zext i32 %102 to i64
$i32B

	full_text


i32 %102
hgetelementptrBW
U
	full_textH
F
D%104 = getelementptr inbounds <4 x float>, <4 x float>* %0, i64 %103
$i64B

	full_text


i64 %103
YstoreBP
N
	full_textA
?
=store <4 x float> %101, <4 x float>* %104, align 16, !tbaa !9
4<4 x float>B#
!
	full_text

<4 x float> %101
6<4 x float>*B$
"
	full_text

<4 x float>* %104
"retB

	full_text


ret void
6<4 x float>*8B"
 
	full_text

<4 x float>* %1
6<4 x float>*8B"
 
	full_text

<4 x float>* %2
6<4 x float>*8B"
 
	full_text

<4 x float>* %0
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
#i328B

	full_text	

i32 3
J	<4 x i32>8B9
7
	full_text*
(
&<4 x i32> <i32 0, i32 1, i32 2, i32 6>
N	<4 x i32>8B=
;
	full_text.
,
*<4 x i32> <i32 0, i32 1, i32 4, i32 undef>
N	<4 x i32>8B=
;
	full_text.
,
*<4 x i32> <i32 0, i32 1, i32 5, i32 undef>
J	<4 x i32>8B9
7
	full_text*
(
&<4 x i32> <i32 0, i32 1, i32 2, i32 5>
J	<4 x i32>8B9
7
	full_text*
(
&<4 x i32> <i32 0, i32 1, i32 2, i32 7>
N	<4 x i32>8B=
;
	full_text.
,
*<4 x i32> <i32 0, i32 1, i32 6, i32 undef>
R	<4 x i32>8BA
?
	full_text2
0
.<4 x i32> <i32 2, i32 6, i32 undef, i32 undef>
J	<4 x i32>8B9
7
	full_text*
(
&<4 x i32> <i32 0, i32 1, i32 2, i32 4>
#i328B

	full_text	

i32 0
#i328B

	full_text	

i32 1
#i328B

	full_text	

i32 2
R	<4 x i32>8BA
?
	full_text2
0
.<4 x i32> <i32 1, i32 5, i32 undef, i32 undef>
R	<4 x i32>8BA
?
	full_text2
0
.<4 x i32> <i32 0, i32 4, i32 undef, i32 undef>
N	<4 x i32>8B=
;
	full_text.
,
*<4 x i32> <i32 0, i32 1, i32 7, i32 undef>
R	<4 x i32>8BA
?
	full_text2
0
.<4 x i32> <i32 3, i32 7, i32 undef, i32 undef>       	  

                       !  "# "$ "" %& %' %% () (( *+ *, ** -. -/ -- 01 00 23 24 22 56 57 55 89 88 :; :: <= << >? >> @A @@ BC BD BB EF EG EE HI HH JK JJ LM LL NO NP NN QR QQ ST SS UV UW UU XY XX Z[ Z\ ZZ ]^ ]] _` __ ab aa cd cc ef eg ee hi hh jk jj lm ln ll op oo qr qs qq tu tt vw vv xy xx z{ zz |} |~ || €  
‚  ƒ„ ƒ
… ƒƒ †† ‡ˆ ‡
‰ ‡‡ Š‹ Š
Œ ŠŠ Ž 
  ‘ 
’  “” ““ •– •
— •• ˜™ ˜
š ˜˜ ›œ ›› 
ž  Ÿ  ŸŸ ¡¢ ¡
£ ¡¡ ¤¥ ¤¤ ¦
§ ¦¦ ¨© ¨¨ ª« ª
¬ ªª ­® ­­ ¯
° ¯¯ ±² ±± ³´ ³
µ ³³ ¶· ¶¶ ¸
¹ ¸¸ º» ºº ¼½ ¼
¾ ¼¼ ¿À ¿
Á ¿¿ ÂÃ Â
Ä ÂÂ ÅÆ ÅÅ Ç
È ÇÇ ÉÊ É
Ë ÉÉ ÌÍ Ì
Î ÌÌ ÏÐ Ï
Ñ ÏÏ ÒÓ Ò
Ô ÒÒ ÕÖ Õ
× ÕÕ ØÙ ØØ Ú
Û ÚÚ ÜÝ Ü
Þ ÜÜ ßà ß
á ßß âã â
ä ââ åæ å
ç åå èé è
ê èè ëì ëë í
î íí ïð ï
ñ ïï òó ò
ô òò õö õ
÷ õõ øù ø
ú øø ûü û
ý ûû þÿ þþ €
 €€ ‚ƒ ‚
„ ‚‚ …† :† J† _† v‡ @‡ S‡ j‡ ‡ ‡ ¦‡ ¯‡ ¸ˆ Çˆ Úˆ íˆ €   	
             ! # $" & ' )( +% , .* / 10 3 42 6 7- 98 ;: =5 ?> A< C@ D- F GE IH KJ M5 O PN RQ TL VS W Y- [X \Z ^] `_ b d5 fc ge ih ka mj n p- ro sq ut wv y {5 }z ~| € ‚x „ …" ˆ ‰ ‹ Œ( ŽŠ ‡ ‘ ’ ”“ – —• ™ š˜ œ› ž  ˜ ¢ £¡ ¥¤ §¦ ©˜ «c ¬ª ®­ °¯ ²˜ ´z µ³ ·¶ ¹¸ »Ÿ ½¨ ¾¼ À± Á¿ Ãº Ä ÆÅ ÈÂ ÊÇ ËŸ Í¨ ÎÌ Ð± ÑÏ Óº Ô Ö ×Õ ÙØ ÛÒ ÝÚ ÞŸ à¨ áß ã± äâ æº ç éX êè ìë îå ðí ñŸ ó¨ ôò ö± ÷õ ùº ú üo ýû ÿþ ø ƒ€ „ … ŠŠ ŒŒ  ŽŽ ‰‰ ‹‹ ŠŠ 
 ‹‹ 
   ŠŠ  ŒŒ † ŽŽ † ŒŒ  ‰‰ 	 o	 z
 å
‘ ¿
’ Ï
“ Ò
” ø
• â
– ß
— Â˜ ˜ ˜ 
˜ ˜ ™ ™ 	™ X	™ c™ †	š (	š 0
š “
› Ì
œ ¼
 õ
ž ò"
matrixTranspose"
_Z15get_global_sizej"
_Z12get_group_idj"
_Z14get_num_groupsj"
_Z12get_local_idj"
_Z14get_local_sizej"
_Z7barrierj*“
MatrixTranspose_Kernels.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02€
 
transfer_bytes_log1p
[&A

devmap_label
 

wgsize_log1p
[&A

transfer_bytes
€€

wgsize
€