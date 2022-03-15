

[external]
JcallBB
@
	full_text3
1
/%9 = tail call i64 @_Z12get_group_idj(i32 1) #4
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
0%11 = tail call i64 @_Z12get_local_idj(i32 0) #4
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
0%13 = tail call i64 @_Z12get_local_idj(i32 1) #4
6truncB-
+
	full_text

%14 = trunc i64 %13 to i32
#i64B

	full_text
	
i64 %13
1shlB*
(
	full_text

%15 = shl nsw i32 %5, 6
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
3icmpB+
)
	full_text

%17 = icmp sgt i32 %5, 0
9brB3
1
	full_text$
"
 br i1 %17, label %18, label %296
!i1B

	full_text


i1 %17
/shl8B&
$
	full_text

%19 = shl i32 %6, 6
5add8B,
*
	full_text

%20 = add nsw i32 %16, %5
%i328B

	full_text
	
i32 %16
Mcall8BC
A
	full_text4
2
0%21 = tail call i64 @_Z12get_group_idj(i32 0) #4
8trunc8B-
+
	full_text

%22 = trunc i64 %21 to i32
%i648B

	full_text
	
i64 %21
4shl8B+
)
	full_text

%23 = shl nsw i32 %22, 6
%i328B

	full_text
	
i32 %22
5mul8B,
*
	full_text

%24 = mul nsw i32 %14, %5
%i328B

	full_text
	
i32 %14
2add8B)
'
	full_text

%25 = add i32 %24, %12
%i328B

	full_text
	
i32 %24
%i328B

	full_text
	
i32 %12
4shl8B+
)
	full_text

%26 = shl nsw i32 %14, 6
%i328B

	full_text
	
i32 %14
6add8B-
+
	full_text

%27 = add nsw i32 %26, %12
%i328B

	full_text
	
i32 %26
%i328B

	full_text
	
i32 %12
6sext8B,
*
	full_text

%28 = sext i32 %27 to i64
%i328B

	full_text
	
i32 %27
\getelementptr8BI
G
	full_text:
8
6%29 = getelementptr inbounds float, float* %3, i64 %28
%i648B

	full_text
	
i64 %28
@bitcast8B3
1
	full_text$
"
 %30 = bitcast float* %29 to i32*
+float*8B

	full_text


float* %29
5mul8B,
*
	full_text

%31 = mul nsw i32 %14, %6
%i328B

	full_text
	
i32 %14
2add8B)
'
	full_text

%32 = add i32 %31, %12
%i328B

	full_text
	
i32 %31
%i328B

	full_text
	
i32 %12
\getelementptr8BI
G
	full_text:
8
6%33 = getelementptr inbounds float, float* %4, i64 %28
%i648B

	full_text
	
i64 %28
@bitcast8B3
1
	full_text$
"
 %34 = bitcast float* %33 to i32*
+float*8B

	full_text


float* %33
6sext8B,
*
	full_text

%35 = sext i32 %26 to i64
%i328B

	full_text
	
i32 %26
1shl8B(
&
	full_text

%36 = shl i64 %11, 32
%i648B

	full_text
	
i64 %11
9ashr8B/
-
	full_text 

%37 = ashr exact i64 %36, 32
%i648B

	full_text
	
i64 %36
\getelementptr8BI
G
	full_text:
8
6%38 = getelementptr inbounds float, float* %3, i64 %35
%i648B

	full_text
	
i64 %35
6sext8B,
*
	full_text

%39 = sext i32 %23 to i64
%i328B

	full_text
	
i32 %23
6sext8B,
*
	full_text

%40 = sext i32 %19 to i64
%i328B

	full_text
	
i32 %19
6sext8B,
*
	full_text

%41 = sext i32 %16 to i64
%i328B

	full_text
	
i32 %16
6sext8B,
*
	full_text

%42 = sext i32 %20 to i64
%i328B

	full_text
	
i32 %20
\getelementptr8BI
G
	full_text:
8
6%43 = getelementptr inbounds float, float* %4, i64 %37
%i648B

	full_text
	
i64 %37
.or8B&
$
	full_text

%44 = or i64 %35, 1
%i648B

	full_text
	
i64 %35
\getelementptr8BI
G
	full_text:
8
6%45 = getelementptr inbounds float, float* %3, i64 %44
%i648B

	full_text
	
i64 %44
5add8B,
*
	full_text

%46 = add nsw i64 %37, 64
%i648B

	full_text
	
i64 %37
\getelementptr8BI
G
	full_text:
8
6%47 = getelementptr inbounds float, float* %4, i64 %46
%i648B

	full_text
	
i64 %46
.or8B&
$
	full_text

%48 = or i64 %35, 2
%i648B

	full_text
	
i64 %35
\getelementptr8BI
G
	full_text:
8
6%49 = getelementptr inbounds float, float* %3, i64 %48
%i648B

	full_text
	
i64 %48
6add8B-
+
	full_text

%50 = add nsw i64 %37, 128
%i648B

	full_text
	
i64 %37
\getelementptr8BI
G
	full_text:
8
6%51 = getelementptr inbounds float, float* %4, i64 %50
%i648B

	full_text
	
i64 %50
.or8B&
$
	full_text

%52 = or i64 %35, 3
%i648B

	full_text
	
i64 %35
\getelementptr8BI
G
	full_text:
8
6%53 = getelementptr inbounds float, float* %3, i64 %52
%i648B

	full_text
	
i64 %52
6add8B-
+
	full_text

%54 = add nsw i64 %37, 192
%i648B

	full_text
	
i64 %37
\getelementptr8BI
G
	full_text:
8
6%55 = getelementptr inbounds float, float* %4, i64 %54
%i648B

	full_text
	
i64 %54
.or8B&
$
	full_text

%56 = or i64 %35, 4
%i648B

	full_text
	
i64 %35
\getelementptr8BI
G
	full_text:
8
6%57 = getelementptr inbounds float, float* %3, i64 %56
%i648B

	full_text
	
i64 %56
6add8B-
+
	full_text

%58 = add nsw i64 %37, 256
%i648B

	full_text
	
i64 %37
\getelementptr8BI
G
	full_text:
8
6%59 = getelementptr inbounds float, float* %4, i64 %58
%i648B

	full_text
	
i64 %58
.or8B&
$
	full_text

%60 = or i64 %35, 5
%i648B

	full_text
	
i64 %35
\getelementptr8BI
G
	full_text:
8
6%61 = getelementptr inbounds float, float* %3, i64 %60
%i648B

	full_text
	
i64 %60
6add8B-
+
	full_text

%62 = add nsw i64 %37, 320
%i648B

	full_text
	
i64 %37
\getelementptr8BI
G
	full_text:
8
6%63 = getelementptr inbounds float, float* %4, i64 %62
%i648B

	full_text
	
i64 %62
.or8B&
$
	full_text

%64 = or i64 %35, 6
%i648B

	full_text
	
i64 %35
\getelementptr8BI
G
	full_text:
8
6%65 = getelementptr inbounds float, float* %3, i64 %64
%i648B

	full_text
	
i64 %64
6add8B-
+
	full_text

%66 = add nsw i64 %37, 384
%i648B

	full_text
	
i64 %37
\getelementptr8BI
G
	full_text:
8
6%67 = getelementptr inbounds float, float* %4, i64 %66
%i648B

	full_text
	
i64 %66
.or8B&
$
	full_text

%68 = or i64 %35, 7
%i648B

	full_text
	
i64 %35
\getelementptr8BI
G
	full_text:
8
6%69 = getelementptr inbounds float, float* %3, i64 %68
%i648B

	full_text
	
i64 %68
6add8B-
+
	full_text

%70 = add nsw i64 %37, 448
%i648B

	full_text
	
i64 %37
\getelementptr8BI
G
	full_text:
8
6%71 = getelementptr inbounds float, float* %4, i64 %70
%i648B

	full_text
	
i64 %70
.or8B&
$
	full_text

%72 = or i64 %35, 8
%i648B

	full_text
	
i64 %35
\getelementptr8BI
G
	full_text:
8
6%73 = getelementptr inbounds float, float* %3, i64 %72
%i648B

	full_text
	
i64 %72
6add8B-
+
	full_text

%74 = add nsw i64 %37, 512
%i648B

	full_text
	
i64 %37
\getelementptr8BI
G
	full_text:
8
6%75 = getelementptr inbounds float, float* %4, i64 %74
%i648B

	full_text
	
i64 %74
.or8B&
$
	full_text

%76 = or i64 %35, 9
%i648B

	full_text
	
i64 %35
\getelementptr8BI
G
	full_text:
8
6%77 = getelementptr inbounds float, float* %3, i64 %76
%i648B

	full_text
	
i64 %76
6add8B-
+
	full_text

%78 = add nsw i64 %37, 576
%i648B

	full_text
	
i64 %37
\getelementptr8BI
G
	full_text:
8
6%79 = getelementptr inbounds float, float* %4, i64 %78
%i648B

	full_text
	
i64 %78
/or8B'
%
	full_text

%80 = or i64 %35, 10
%i648B

	full_text
	
i64 %35
\getelementptr8BI
G
	full_text:
8
6%81 = getelementptr inbounds float, float* %3, i64 %80
%i648B

	full_text
	
i64 %80
6add8B-
+
	full_text

%82 = add nsw i64 %37, 640
%i648B

	full_text
	
i64 %37
\getelementptr8BI
G
	full_text:
8
6%83 = getelementptr inbounds float, float* %4, i64 %82
%i648B

	full_text
	
i64 %82
/or8B'
%
	full_text

%84 = or i64 %35, 11
%i648B

	full_text
	
i64 %35
\getelementptr8BI
G
	full_text:
8
6%85 = getelementptr inbounds float, float* %3, i64 %84
%i648B

	full_text
	
i64 %84
6add8B-
+
	full_text

%86 = add nsw i64 %37, 704
%i648B

	full_text
	
i64 %37
\getelementptr8BI
G
	full_text:
8
6%87 = getelementptr inbounds float, float* %4, i64 %86
%i648B

	full_text
	
i64 %86
/or8B'
%
	full_text

%88 = or i64 %35, 12
%i648B

	full_text
	
i64 %35
\getelementptr8BI
G
	full_text:
8
6%89 = getelementptr inbounds float, float* %3, i64 %88
%i648B

	full_text
	
i64 %88
6add8B-
+
	full_text

%90 = add nsw i64 %37, 768
%i648B

	full_text
	
i64 %37
\getelementptr8BI
G
	full_text:
8
6%91 = getelementptr inbounds float, float* %4, i64 %90
%i648B

	full_text
	
i64 %90
/or8B'
%
	full_text

%92 = or i64 %35, 13
%i648B

	full_text
	
i64 %35
\getelementptr8BI
G
	full_text:
8
6%93 = getelementptr inbounds float, float* %3, i64 %92
%i648B

	full_text
	
i64 %92
6add8B-
+
	full_text

%94 = add nsw i64 %37, 832
%i648B

	full_text
	
i64 %37
\getelementptr8BI
G
	full_text:
8
6%95 = getelementptr inbounds float, float* %4, i64 %94
%i648B

	full_text
	
i64 %94
/or8B'
%
	full_text

%96 = or i64 %35, 14
%i648B

	full_text
	
i64 %35
\getelementptr8BI
G
	full_text:
8
6%97 = getelementptr inbounds float, float* %3, i64 %96
%i648B

	full_text
	
i64 %96
6add8B-
+
	full_text

%98 = add nsw i64 %37, 896
%i648B

	full_text
	
i64 %37
\getelementptr8BI
G
	full_text:
8
6%99 = getelementptr inbounds float, float* %4, i64 %98
%i648B

	full_text
	
i64 %98
0or8B(
&
	full_text

%100 = or i64 %35, 15
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%101 = getelementptr inbounds float, float* %3, i64 %100
&i648B

	full_text


i64 %100
7add8B.
,
	full_text

%102 = add nsw i64 %37, 960
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%103 = getelementptr inbounds float, float* %4, i64 %102
&i648B

	full_text


i64 %102
0or8B(
&
	full_text

%104 = or i64 %35, 16
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%105 = getelementptr inbounds float, float* %3, i64 %104
&i648B

	full_text


i64 %104
8add8B/
-
	full_text 

%106 = add nsw i64 %37, 1024
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%107 = getelementptr inbounds float, float* %4, i64 %106
&i648B

	full_text


i64 %106
0or8B(
&
	full_text

%108 = or i64 %35, 17
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%109 = getelementptr inbounds float, float* %3, i64 %108
&i648B

	full_text


i64 %108
8add8B/
-
	full_text 

%110 = add nsw i64 %37, 1088
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%111 = getelementptr inbounds float, float* %4, i64 %110
&i648B

	full_text


i64 %110
0or8B(
&
	full_text

%112 = or i64 %35, 18
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%113 = getelementptr inbounds float, float* %3, i64 %112
&i648B

	full_text


i64 %112
8add8B/
-
	full_text 

%114 = add nsw i64 %37, 1152
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%115 = getelementptr inbounds float, float* %4, i64 %114
&i648B

	full_text


i64 %114
0or8B(
&
	full_text

%116 = or i64 %35, 19
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%117 = getelementptr inbounds float, float* %3, i64 %116
&i648B

	full_text


i64 %116
8add8B/
-
	full_text 

%118 = add nsw i64 %37, 1216
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%119 = getelementptr inbounds float, float* %4, i64 %118
&i648B

	full_text


i64 %118
0or8B(
&
	full_text

%120 = or i64 %35, 20
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%121 = getelementptr inbounds float, float* %3, i64 %120
&i648B

	full_text


i64 %120
8add8B/
-
	full_text 

%122 = add nsw i64 %37, 1280
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%123 = getelementptr inbounds float, float* %4, i64 %122
&i648B

	full_text


i64 %122
0or8B(
&
	full_text

%124 = or i64 %35, 21
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%125 = getelementptr inbounds float, float* %3, i64 %124
&i648B

	full_text


i64 %124
8add8B/
-
	full_text 

%126 = add nsw i64 %37, 1344
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%127 = getelementptr inbounds float, float* %4, i64 %126
&i648B

	full_text


i64 %126
0or8B(
&
	full_text

%128 = or i64 %35, 22
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%129 = getelementptr inbounds float, float* %3, i64 %128
&i648B

	full_text


i64 %128
8add8B/
-
	full_text 

%130 = add nsw i64 %37, 1408
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%131 = getelementptr inbounds float, float* %4, i64 %130
&i648B

	full_text


i64 %130
0or8B(
&
	full_text

%132 = or i64 %35, 23
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%133 = getelementptr inbounds float, float* %3, i64 %132
&i648B

	full_text


i64 %132
8add8B/
-
	full_text 

%134 = add nsw i64 %37, 1472
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%135 = getelementptr inbounds float, float* %4, i64 %134
&i648B

	full_text


i64 %134
0or8B(
&
	full_text

%136 = or i64 %35, 24
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%137 = getelementptr inbounds float, float* %3, i64 %136
&i648B

	full_text


i64 %136
8add8B/
-
	full_text 

%138 = add nsw i64 %37, 1536
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%139 = getelementptr inbounds float, float* %4, i64 %138
&i648B

	full_text


i64 %138
0or8B(
&
	full_text

%140 = or i64 %35, 25
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%141 = getelementptr inbounds float, float* %3, i64 %140
&i648B

	full_text


i64 %140
8add8B/
-
	full_text 

%142 = add nsw i64 %37, 1600
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%143 = getelementptr inbounds float, float* %4, i64 %142
&i648B

	full_text


i64 %142
0or8B(
&
	full_text

%144 = or i64 %35, 26
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%145 = getelementptr inbounds float, float* %3, i64 %144
&i648B

	full_text


i64 %144
8add8B/
-
	full_text 

%146 = add nsw i64 %37, 1664
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%147 = getelementptr inbounds float, float* %4, i64 %146
&i648B

	full_text


i64 %146
0or8B(
&
	full_text

%148 = or i64 %35, 27
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%149 = getelementptr inbounds float, float* %3, i64 %148
&i648B

	full_text


i64 %148
8add8B/
-
	full_text 

%150 = add nsw i64 %37, 1728
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%151 = getelementptr inbounds float, float* %4, i64 %150
&i648B

	full_text


i64 %150
0or8B(
&
	full_text

%152 = or i64 %35, 28
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%153 = getelementptr inbounds float, float* %3, i64 %152
&i648B

	full_text


i64 %152
8add8B/
-
	full_text 

%154 = add nsw i64 %37, 1792
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%155 = getelementptr inbounds float, float* %4, i64 %154
&i648B

	full_text


i64 %154
0or8B(
&
	full_text

%156 = or i64 %35, 29
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%157 = getelementptr inbounds float, float* %3, i64 %156
&i648B

	full_text


i64 %156
8add8B/
-
	full_text 

%158 = add nsw i64 %37, 1856
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%159 = getelementptr inbounds float, float* %4, i64 %158
&i648B

	full_text


i64 %158
0or8B(
&
	full_text

%160 = or i64 %35, 30
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%161 = getelementptr inbounds float, float* %3, i64 %160
&i648B

	full_text


i64 %160
8add8B/
-
	full_text 

%162 = add nsw i64 %37, 1920
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%163 = getelementptr inbounds float, float* %4, i64 %162
&i648B

	full_text


i64 %162
0or8B(
&
	full_text

%164 = or i64 %35, 31
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%165 = getelementptr inbounds float, float* %3, i64 %164
&i648B

	full_text


i64 %164
8add8B/
-
	full_text 

%166 = add nsw i64 %37, 1984
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%167 = getelementptr inbounds float, float* %4, i64 %166
&i648B

	full_text


i64 %166
0or8B(
&
	full_text

%168 = or i64 %35, 32
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%169 = getelementptr inbounds float, float* %3, i64 %168
&i648B

	full_text


i64 %168
8add8B/
-
	full_text 

%170 = add nsw i64 %37, 2048
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%171 = getelementptr inbounds float, float* %4, i64 %170
&i648B

	full_text


i64 %170
0or8B(
&
	full_text

%172 = or i64 %35, 33
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%173 = getelementptr inbounds float, float* %3, i64 %172
&i648B

	full_text


i64 %172
8add8B/
-
	full_text 

%174 = add nsw i64 %37, 2112
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%175 = getelementptr inbounds float, float* %4, i64 %174
&i648B

	full_text


i64 %174
0or8B(
&
	full_text

%176 = or i64 %35, 34
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%177 = getelementptr inbounds float, float* %3, i64 %176
&i648B

	full_text


i64 %176
8add8B/
-
	full_text 

%178 = add nsw i64 %37, 2176
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%179 = getelementptr inbounds float, float* %4, i64 %178
&i648B

	full_text


i64 %178
0or8B(
&
	full_text

%180 = or i64 %35, 35
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%181 = getelementptr inbounds float, float* %3, i64 %180
&i648B

	full_text


i64 %180
8add8B/
-
	full_text 

%182 = add nsw i64 %37, 2240
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%183 = getelementptr inbounds float, float* %4, i64 %182
&i648B

	full_text


i64 %182
0or8B(
&
	full_text

%184 = or i64 %35, 36
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%185 = getelementptr inbounds float, float* %3, i64 %184
&i648B

	full_text


i64 %184
8add8B/
-
	full_text 

%186 = add nsw i64 %37, 2304
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%187 = getelementptr inbounds float, float* %4, i64 %186
&i648B

	full_text


i64 %186
0or8B(
&
	full_text

%188 = or i64 %35, 37
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%189 = getelementptr inbounds float, float* %3, i64 %188
&i648B

	full_text


i64 %188
8add8B/
-
	full_text 

%190 = add nsw i64 %37, 2368
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%191 = getelementptr inbounds float, float* %4, i64 %190
&i648B

	full_text


i64 %190
0or8B(
&
	full_text

%192 = or i64 %35, 38
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%193 = getelementptr inbounds float, float* %3, i64 %192
&i648B

	full_text


i64 %192
8add8B/
-
	full_text 

%194 = add nsw i64 %37, 2432
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%195 = getelementptr inbounds float, float* %4, i64 %194
&i648B

	full_text


i64 %194
0or8B(
&
	full_text

%196 = or i64 %35, 39
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%197 = getelementptr inbounds float, float* %3, i64 %196
&i648B

	full_text


i64 %196
8add8B/
-
	full_text 

%198 = add nsw i64 %37, 2496
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%199 = getelementptr inbounds float, float* %4, i64 %198
&i648B

	full_text


i64 %198
0or8B(
&
	full_text

%200 = or i64 %35, 40
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%201 = getelementptr inbounds float, float* %3, i64 %200
&i648B

	full_text


i64 %200
8add8B/
-
	full_text 

%202 = add nsw i64 %37, 2560
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%203 = getelementptr inbounds float, float* %4, i64 %202
&i648B

	full_text


i64 %202
0or8B(
&
	full_text

%204 = or i64 %35, 41
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%205 = getelementptr inbounds float, float* %3, i64 %204
&i648B

	full_text


i64 %204
8add8B/
-
	full_text 

%206 = add nsw i64 %37, 2624
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%207 = getelementptr inbounds float, float* %4, i64 %206
&i648B

	full_text


i64 %206
0or8B(
&
	full_text

%208 = or i64 %35, 42
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%209 = getelementptr inbounds float, float* %3, i64 %208
&i648B

	full_text


i64 %208
8add8B/
-
	full_text 

%210 = add nsw i64 %37, 2688
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%211 = getelementptr inbounds float, float* %4, i64 %210
&i648B

	full_text


i64 %210
0or8B(
&
	full_text

%212 = or i64 %35, 43
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%213 = getelementptr inbounds float, float* %3, i64 %212
&i648B

	full_text


i64 %212
8add8B/
-
	full_text 

%214 = add nsw i64 %37, 2752
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%215 = getelementptr inbounds float, float* %4, i64 %214
&i648B

	full_text


i64 %214
0or8B(
&
	full_text

%216 = or i64 %35, 44
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%217 = getelementptr inbounds float, float* %3, i64 %216
&i648B

	full_text


i64 %216
8add8B/
-
	full_text 

%218 = add nsw i64 %37, 2816
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%219 = getelementptr inbounds float, float* %4, i64 %218
&i648B

	full_text


i64 %218
0or8B(
&
	full_text

%220 = or i64 %35, 45
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%221 = getelementptr inbounds float, float* %3, i64 %220
&i648B

	full_text


i64 %220
8add8B/
-
	full_text 

%222 = add nsw i64 %37, 2880
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%223 = getelementptr inbounds float, float* %4, i64 %222
&i648B

	full_text


i64 %222
0or8B(
&
	full_text

%224 = or i64 %35, 46
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%225 = getelementptr inbounds float, float* %3, i64 %224
&i648B

	full_text


i64 %224
8add8B/
-
	full_text 

%226 = add nsw i64 %37, 2944
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%227 = getelementptr inbounds float, float* %4, i64 %226
&i648B

	full_text


i64 %226
0or8B(
&
	full_text

%228 = or i64 %35, 47
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%229 = getelementptr inbounds float, float* %3, i64 %228
&i648B

	full_text


i64 %228
8add8B/
-
	full_text 

%230 = add nsw i64 %37, 3008
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%231 = getelementptr inbounds float, float* %4, i64 %230
&i648B

	full_text


i64 %230
0or8B(
&
	full_text

%232 = or i64 %35, 48
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%233 = getelementptr inbounds float, float* %3, i64 %232
&i648B

	full_text


i64 %232
8add8B/
-
	full_text 

%234 = add nsw i64 %37, 3072
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%235 = getelementptr inbounds float, float* %4, i64 %234
&i648B

	full_text


i64 %234
0or8B(
&
	full_text

%236 = or i64 %35, 49
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%237 = getelementptr inbounds float, float* %3, i64 %236
&i648B

	full_text


i64 %236
8add8B/
-
	full_text 

%238 = add nsw i64 %37, 3136
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%239 = getelementptr inbounds float, float* %4, i64 %238
&i648B

	full_text


i64 %238
0or8B(
&
	full_text

%240 = or i64 %35, 50
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%241 = getelementptr inbounds float, float* %3, i64 %240
&i648B

	full_text


i64 %240
8add8B/
-
	full_text 

%242 = add nsw i64 %37, 3200
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%243 = getelementptr inbounds float, float* %4, i64 %242
&i648B

	full_text


i64 %242
0or8B(
&
	full_text

%244 = or i64 %35, 51
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%245 = getelementptr inbounds float, float* %3, i64 %244
&i648B

	full_text


i64 %244
8add8B/
-
	full_text 

%246 = add nsw i64 %37, 3264
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%247 = getelementptr inbounds float, float* %4, i64 %246
&i648B

	full_text


i64 %246
0or8B(
&
	full_text

%248 = or i64 %35, 52
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%249 = getelementptr inbounds float, float* %3, i64 %248
&i648B

	full_text


i64 %248
8add8B/
-
	full_text 

%250 = add nsw i64 %37, 3328
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%251 = getelementptr inbounds float, float* %4, i64 %250
&i648B

	full_text


i64 %250
0or8B(
&
	full_text

%252 = or i64 %35, 53
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%253 = getelementptr inbounds float, float* %3, i64 %252
&i648B

	full_text


i64 %252
8add8B/
-
	full_text 

%254 = add nsw i64 %37, 3392
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%255 = getelementptr inbounds float, float* %4, i64 %254
&i648B

	full_text


i64 %254
0or8B(
&
	full_text

%256 = or i64 %35, 54
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%257 = getelementptr inbounds float, float* %3, i64 %256
&i648B

	full_text


i64 %256
8add8B/
-
	full_text 

%258 = add nsw i64 %37, 3456
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%259 = getelementptr inbounds float, float* %4, i64 %258
&i648B

	full_text


i64 %258
0or8B(
&
	full_text

%260 = or i64 %35, 55
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%261 = getelementptr inbounds float, float* %3, i64 %260
&i648B

	full_text


i64 %260
8add8B/
-
	full_text 

%262 = add nsw i64 %37, 3520
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%263 = getelementptr inbounds float, float* %4, i64 %262
&i648B

	full_text


i64 %262
0or8B(
&
	full_text

%264 = or i64 %35, 56
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%265 = getelementptr inbounds float, float* %3, i64 %264
&i648B

	full_text


i64 %264
8add8B/
-
	full_text 

%266 = add nsw i64 %37, 3584
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%267 = getelementptr inbounds float, float* %4, i64 %266
&i648B

	full_text


i64 %266
0or8B(
&
	full_text

%268 = or i64 %35, 57
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%269 = getelementptr inbounds float, float* %3, i64 %268
&i648B

	full_text


i64 %268
8add8B/
-
	full_text 

%270 = add nsw i64 %37, 3648
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%271 = getelementptr inbounds float, float* %4, i64 %270
&i648B

	full_text


i64 %270
0or8B(
&
	full_text

%272 = or i64 %35, 58
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%273 = getelementptr inbounds float, float* %3, i64 %272
&i648B

	full_text


i64 %272
8add8B/
-
	full_text 

%274 = add nsw i64 %37, 3712
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%275 = getelementptr inbounds float, float* %4, i64 %274
&i648B

	full_text


i64 %274
0or8B(
&
	full_text

%276 = or i64 %35, 59
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%277 = getelementptr inbounds float, float* %3, i64 %276
&i648B

	full_text


i64 %276
8add8B/
-
	full_text 

%278 = add nsw i64 %37, 3776
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%279 = getelementptr inbounds float, float* %4, i64 %278
&i648B

	full_text


i64 %278
0or8B(
&
	full_text

%280 = or i64 %35, 60
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%281 = getelementptr inbounds float, float* %3, i64 %280
&i648B

	full_text


i64 %280
8add8B/
-
	full_text 

%282 = add nsw i64 %37, 3840
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%283 = getelementptr inbounds float, float* %4, i64 %282
&i648B

	full_text


i64 %282
0or8B(
&
	full_text

%284 = or i64 %35, 61
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%285 = getelementptr inbounds float, float* %3, i64 %284
&i648B

	full_text


i64 %284
8add8B/
-
	full_text 

%286 = add nsw i64 %37, 3904
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%287 = getelementptr inbounds float, float* %4, i64 %286
&i648B

	full_text


i64 %286
0or8B(
&
	full_text

%288 = or i64 %35, 62
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%289 = getelementptr inbounds float, float* %3, i64 %288
&i648B

	full_text


i64 %288
8add8B/
-
	full_text 

%290 = add nsw i64 %37, 3968
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%291 = getelementptr inbounds float, float* %4, i64 %290
&i648B

	full_text


i64 %290
0or8B(
&
	full_text

%292 = or i64 %35, 63
%i648B

	full_text
	
i64 %35
^getelementptr8BK
I
	full_text<
:
8%293 = getelementptr inbounds float, float* %3, i64 %292
&i648B

	full_text


i64 %292
8add8B/
-
	full_text 

%294 = add nsw i64 %37, 4032
%i648B

	full_text
	
i64 %37
^getelementptr8BK
I
	full_text<
:
8%295 = getelementptr inbounds float, float* %4, i64 %294
&i648B

	full_text


i64 %294
(br8B 

	full_text

br label %301
Qphi8BH
F
	full_text9
7
5%297 = phi float [ 0.000000e+00, %8 ], [ %508, %301 ]
*float8B

	full_text


float %508
Ocall8BE
C
	full_text6
4
2%298 = tail call i64 @_Z13get_global_idj(i32 1) #4
6sext8B,
*
	full_text

%299 = sext i32 %7 to i64
;icmp8B1
/
	full_text"
 
%300 = icmp ult i64 %298, %299
&i648B

	full_text


i64 %298
&i648B

	full_text


i64 %299
=br8B5
3
	full_text&
$
"br i1 %300, label %512, label %518
$i18B

	full_text
	
i1 %300
Gphi8B>
<
	full_text/
-
+%302 = phi i64 [ %41, %18 ], [ %509, %301 ]
%i648B

	full_text
	
i64 %41
&i648B

	full_text


i64 %509
Gphi8B>
<
	full_text/
-
+%303 = phi i64 [ %39, %18 ], [ %510, %301 ]
%i648B

	full_text
	
i64 %39
&i648B

	full_text


i64 %510
Rphi8BI
G
	full_text:
8
6%304 = phi float [ 0.000000e+00, %18 ], [ %508, %301 ]
*float8B

	full_text


float %508
:trunc8B/
-
	full_text 

%305 = trunc i64 %302 to i32
&i648B

	full_text


i64 %302
4add8B+
)
	full_text

%306 = add i32 %25, %305
%i328B

	full_text
	
i32 %25
&i328B

	full_text


i32 %305
8sext8B.
,
	full_text

%307 = sext i32 %306 to i64
&i328B

	full_text


i32 %306
^getelementptr8BK
I
	full_text<
:
8%308 = getelementptr inbounds float, float* %1, i64 %307
&i648B

	full_text


i64 %307
Bbitcast8B5
3
	full_text&
$
"%309 = bitcast float* %308 to i32*
,float*8B

	full_text

float* %308
Jload8B@
>
	full_text1
/
-%310 = load i32, i32* %309, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %309
Istore8B>
<
	full_text/
-
+store i32 %310, i32* %30, align 4, !tbaa !8
&i328B

	full_text


i32 %310
'i32*8B

	full_text


i32* %30
:trunc8B/
-
	full_text 

%311 = trunc i64 %303 to i32
&i648B

	full_text


i64 %303
4add8B+
)
	full_text

%312 = add i32 %32, %311
%i328B

	full_text
	
i32 %32
&i328B

	full_text


i32 %311
8sext8B.
,
	full_text

%313 = sext i32 %312 to i64
&i328B

	full_text


i32 %312
^getelementptr8BK
I
	full_text<
:
8%314 = getelementptr inbounds float, float* %2, i64 %313
&i648B

	full_text


i64 %313
Bbitcast8B5
3
	full_text&
$
"%315 = bitcast float* %314 to i32*
,float*8B

	full_text

float* %314
Jload8B@
>
	full_text1
/
-%316 = load i32, i32* %315, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %315
Istore8B>
<
	full_text/
-
+store i32 %316, i32* %34, align 4, !tbaa !8
&i328B

	full_text


i32 %316
'i32*8B

	full_text


i32* %34
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
Mload8BC
A
	full_text4
2
0%317 = load float, float* %38, align 4, !tbaa !8
+float*8B

	full_text


float* %38
Mload8BC
A
	full_text4
2
0%318 = load float, float* %43, align 4, !tbaa !8
+float*8B

	full_text


float* %43
icall8B_
]
	full_textP
N
L%319 = tail call float @llvm.fmuladd.f32(float %317, float %318, float %304)
*float8B

	full_text


float %317
*float8B

	full_text


float %318
*float8B

	full_text


float %304
Mload8BC
A
	full_text4
2
0%320 = load float, float* %45, align 4, !tbaa !8
+float*8B

	full_text


float* %45
Mload8BC
A
	full_text4
2
0%321 = load float, float* %47, align 4, !tbaa !8
+float*8B

	full_text


float* %47
icall8B_
]
	full_textP
N
L%322 = tail call float @llvm.fmuladd.f32(float %320, float %321, float %319)
*float8B

	full_text


float %320
*float8B

	full_text


float %321
*float8B

	full_text


float %319
Mload8BC
A
	full_text4
2
0%323 = load float, float* %49, align 4, !tbaa !8
+float*8B

	full_text


float* %49
Mload8BC
A
	full_text4
2
0%324 = load float, float* %51, align 4, !tbaa !8
+float*8B

	full_text


float* %51
icall8B_
]
	full_textP
N
L%325 = tail call float @llvm.fmuladd.f32(float %323, float %324, float %322)
*float8B

	full_text


float %323
*float8B

	full_text


float %324
*float8B

	full_text


float %322
Mload8BC
A
	full_text4
2
0%326 = load float, float* %53, align 4, !tbaa !8
+float*8B

	full_text


float* %53
Mload8BC
A
	full_text4
2
0%327 = load float, float* %55, align 4, !tbaa !8
+float*8B

	full_text


float* %55
icall8B_
]
	full_textP
N
L%328 = tail call float @llvm.fmuladd.f32(float %326, float %327, float %325)
*float8B

	full_text


float %326
*float8B

	full_text


float %327
*float8B

	full_text


float %325
Mload8BC
A
	full_text4
2
0%329 = load float, float* %57, align 4, !tbaa !8
+float*8B

	full_text


float* %57
Mload8BC
A
	full_text4
2
0%330 = load float, float* %59, align 4, !tbaa !8
+float*8B

	full_text


float* %59
icall8B_
]
	full_textP
N
L%331 = tail call float @llvm.fmuladd.f32(float %329, float %330, float %328)
*float8B

	full_text


float %329
*float8B

	full_text


float %330
*float8B

	full_text


float %328
Mload8BC
A
	full_text4
2
0%332 = load float, float* %61, align 4, !tbaa !8
+float*8B

	full_text


float* %61
Mload8BC
A
	full_text4
2
0%333 = load float, float* %63, align 4, !tbaa !8
+float*8B

	full_text


float* %63
icall8B_
]
	full_textP
N
L%334 = tail call float @llvm.fmuladd.f32(float %332, float %333, float %331)
*float8B

	full_text


float %332
*float8B

	full_text


float %333
*float8B

	full_text


float %331
Mload8BC
A
	full_text4
2
0%335 = load float, float* %65, align 4, !tbaa !8
+float*8B

	full_text


float* %65
Mload8BC
A
	full_text4
2
0%336 = load float, float* %67, align 4, !tbaa !8
+float*8B

	full_text


float* %67
icall8B_
]
	full_textP
N
L%337 = tail call float @llvm.fmuladd.f32(float %335, float %336, float %334)
*float8B

	full_text


float %335
*float8B

	full_text


float %336
*float8B

	full_text


float %334
Mload8BC
A
	full_text4
2
0%338 = load float, float* %69, align 4, !tbaa !8
+float*8B

	full_text


float* %69
Mload8BC
A
	full_text4
2
0%339 = load float, float* %71, align 4, !tbaa !8
+float*8B

	full_text


float* %71
icall8B_
]
	full_textP
N
L%340 = tail call float @llvm.fmuladd.f32(float %338, float %339, float %337)
*float8B

	full_text


float %338
*float8B

	full_text


float %339
*float8B

	full_text


float %337
Mload8BC
A
	full_text4
2
0%341 = load float, float* %73, align 4, !tbaa !8
+float*8B

	full_text


float* %73
Mload8BC
A
	full_text4
2
0%342 = load float, float* %75, align 4, !tbaa !8
+float*8B

	full_text


float* %75
icall8B_
]
	full_textP
N
L%343 = tail call float @llvm.fmuladd.f32(float %341, float %342, float %340)
*float8B

	full_text


float %341
*float8B

	full_text


float %342
*float8B

	full_text


float %340
Mload8BC
A
	full_text4
2
0%344 = load float, float* %77, align 4, !tbaa !8
+float*8B

	full_text


float* %77
Mload8BC
A
	full_text4
2
0%345 = load float, float* %79, align 4, !tbaa !8
+float*8B

	full_text


float* %79
icall8B_
]
	full_textP
N
L%346 = tail call float @llvm.fmuladd.f32(float %344, float %345, float %343)
*float8B

	full_text


float %344
*float8B

	full_text


float %345
*float8B

	full_text


float %343
Mload8BC
A
	full_text4
2
0%347 = load float, float* %81, align 4, !tbaa !8
+float*8B

	full_text


float* %81
Mload8BC
A
	full_text4
2
0%348 = load float, float* %83, align 4, !tbaa !8
+float*8B

	full_text


float* %83
icall8B_
]
	full_textP
N
L%349 = tail call float @llvm.fmuladd.f32(float %347, float %348, float %346)
*float8B

	full_text


float %347
*float8B

	full_text


float %348
*float8B

	full_text


float %346
Mload8BC
A
	full_text4
2
0%350 = load float, float* %85, align 4, !tbaa !8
+float*8B

	full_text


float* %85
Mload8BC
A
	full_text4
2
0%351 = load float, float* %87, align 4, !tbaa !8
+float*8B

	full_text


float* %87
icall8B_
]
	full_textP
N
L%352 = tail call float @llvm.fmuladd.f32(float %350, float %351, float %349)
*float8B

	full_text


float %350
*float8B

	full_text


float %351
*float8B

	full_text


float %349
Mload8BC
A
	full_text4
2
0%353 = load float, float* %89, align 4, !tbaa !8
+float*8B

	full_text


float* %89
Mload8BC
A
	full_text4
2
0%354 = load float, float* %91, align 4, !tbaa !8
+float*8B

	full_text


float* %91
icall8B_
]
	full_textP
N
L%355 = tail call float @llvm.fmuladd.f32(float %353, float %354, float %352)
*float8B

	full_text


float %353
*float8B

	full_text


float %354
*float8B

	full_text


float %352
Mload8BC
A
	full_text4
2
0%356 = load float, float* %93, align 4, !tbaa !8
+float*8B

	full_text


float* %93
Mload8BC
A
	full_text4
2
0%357 = load float, float* %95, align 4, !tbaa !8
+float*8B

	full_text


float* %95
icall8B_
]
	full_textP
N
L%358 = tail call float @llvm.fmuladd.f32(float %356, float %357, float %355)
*float8B

	full_text


float %356
*float8B

	full_text


float %357
*float8B

	full_text


float %355
Mload8BC
A
	full_text4
2
0%359 = load float, float* %97, align 4, !tbaa !8
+float*8B

	full_text


float* %97
Mload8BC
A
	full_text4
2
0%360 = load float, float* %99, align 4, !tbaa !8
+float*8B

	full_text


float* %99
icall8B_
]
	full_textP
N
L%361 = tail call float @llvm.fmuladd.f32(float %359, float %360, float %358)
*float8B

	full_text


float %359
*float8B

	full_text


float %360
*float8B

	full_text


float %358
Nload8BD
B
	full_text5
3
1%362 = load float, float* %101, align 4, !tbaa !8
,float*8B

	full_text

float* %101
Nload8BD
B
	full_text5
3
1%363 = load float, float* %103, align 4, !tbaa !8
,float*8B

	full_text

float* %103
icall8B_
]
	full_textP
N
L%364 = tail call float @llvm.fmuladd.f32(float %362, float %363, float %361)
*float8B

	full_text


float %362
*float8B

	full_text


float %363
*float8B

	full_text


float %361
Nload8BD
B
	full_text5
3
1%365 = load float, float* %105, align 4, !tbaa !8
,float*8B

	full_text

float* %105
Nload8BD
B
	full_text5
3
1%366 = load float, float* %107, align 4, !tbaa !8
,float*8B

	full_text

float* %107
icall8B_
]
	full_textP
N
L%367 = tail call float @llvm.fmuladd.f32(float %365, float %366, float %364)
*float8B

	full_text


float %365
*float8B

	full_text


float %366
*float8B

	full_text


float %364
Nload8BD
B
	full_text5
3
1%368 = load float, float* %109, align 4, !tbaa !8
,float*8B

	full_text

float* %109
Nload8BD
B
	full_text5
3
1%369 = load float, float* %111, align 4, !tbaa !8
,float*8B

	full_text

float* %111
icall8B_
]
	full_textP
N
L%370 = tail call float @llvm.fmuladd.f32(float %368, float %369, float %367)
*float8B

	full_text


float %368
*float8B

	full_text


float %369
*float8B

	full_text


float %367
Nload8BD
B
	full_text5
3
1%371 = load float, float* %113, align 4, !tbaa !8
,float*8B

	full_text

float* %113
Nload8BD
B
	full_text5
3
1%372 = load float, float* %115, align 4, !tbaa !8
,float*8B

	full_text

float* %115
icall8B_
]
	full_textP
N
L%373 = tail call float @llvm.fmuladd.f32(float %371, float %372, float %370)
*float8B

	full_text


float %371
*float8B

	full_text


float %372
*float8B

	full_text


float %370
Nload8BD
B
	full_text5
3
1%374 = load float, float* %117, align 4, !tbaa !8
,float*8B

	full_text

float* %117
Nload8BD
B
	full_text5
3
1%375 = load float, float* %119, align 4, !tbaa !8
,float*8B

	full_text

float* %119
icall8B_
]
	full_textP
N
L%376 = tail call float @llvm.fmuladd.f32(float %374, float %375, float %373)
*float8B

	full_text


float %374
*float8B

	full_text


float %375
*float8B

	full_text


float %373
Nload8BD
B
	full_text5
3
1%377 = load float, float* %121, align 4, !tbaa !8
,float*8B

	full_text

float* %121
Nload8BD
B
	full_text5
3
1%378 = load float, float* %123, align 4, !tbaa !8
,float*8B

	full_text

float* %123
icall8B_
]
	full_textP
N
L%379 = tail call float @llvm.fmuladd.f32(float %377, float %378, float %376)
*float8B

	full_text


float %377
*float8B

	full_text


float %378
*float8B

	full_text


float %376
Nload8BD
B
	full_text5
3
1%380 = load float, float* %125, align 4, !tbaa !8
,float*8B

	full_text

float* %125
Nload8BD
B
	full_text5
3
1%381 = load float, float* %127, align 4, !tbaa !8
,float*8B

	full_text

float* %127
icall8B_
]
	full_textP
N
L%382 = tail call float @llvm.fmuladd.f32(float %380, float %381, float %379)
*float8B

	full_text


float %380
*float8B

	full_text


float %381
*float8B

	full_text


float %379
Nload8BD
B
	full_text5
3
1%383 = load float, float* %129, align 4, !tbaa !8
,float*8B

	full_text

float* %129
Nload8BD
B
	full_text5
3
1%384 = load float, float* %131, align 4, !tbaa !8
,float*8B

	full_text

float* %131
icall8B_
]
	full_textP
N
L%385 = tail call float @llvm.fmuladd.f32(float %383, float %384, float %382)
*float8B

	full_text


float %383
*float8B

	full_text


float %384
*float8B

	full_text


float %382
Nload8BD
B
	full_text5
3
1%386 = load float, float* %133, align 4, !tbaa !8
,float*8B

	full_text

float* %133
Nload8BD
B
	full_text5
3
1%387 = load float, float* %135, align 4, !tbaa !8
,float*8B

	full_text

float* %135
icall8B_
]
	full_textP
N
L%388 = tail call float @llvm.fmuladd.f32(float %386, float %387, float %385)
*float8B

	full_text


float %386
*float8B

	full_text


float %387
*float8B

	full_text


float %385
Nload8BD
B
	full_text5
3
1%389 = load float, float* %137, align 4, !tbaa !8
,float*8B

	full_text

float* %137
Nload8BD
B
	full_text5
3
1%390 = load float, float* %139, align 4, !tbaa !8
,float*8B

	full_text

float* %139
icall8B_
]
	full_textP
N
L%391 = tail call float @llvm.fmuladd.f32(float %389, float %390, float %388)
*float8B

	full_text


float %389
*float8B

	full_text


float %390
*float8B

	full_text


float %388
Nload8BD
B
	full_text5
3
1%392 = load float, float* %141, align 4, !tbaa !8
,float*8B

	full_text

float* %141
Nload8BD
B
	full_text5
3
1%393 = load float, float* %143, align 4, !tbaa !8
,float*8B

	full_text

float* %143
icall8B_
]
	full_textP
N
L%394 = tail call float @llvm.fmuladd.f32(float %392, float %393, float %391)
*float8B

	full_text


float %392
*float8B

	full_text


float %393
*float8B

	full_text


float %391
Nload8BD
B
	full_text5
3
1%395 = load float, float* %145, align 4, !tbaa !8
,float*8B

	full_text

float* %145
Nload8BD
B
	full_text5
3
1%396 = load float, float* %147, align 4, !tbaa !8
,float*8B

	full_text

float* %147
icall8B_
]
	full_textP
N
L%397 = tail call float @llvm.fmuladd.f32(float %395, float %396, float %394)
*float8B

	full_text


float %395
*float8B

	full_text


float %396
*float8B

	full_text


float %394
Nload8BD
B
	full_text5
3
1%398 = load float, float* %149, align 4, !tbaa !8
,float*8B

	full_text

float* %149
Nload8BD
B
	full_text5
3
1%399 = load float, float* %151, align 4, !tbaa !8
,float*8B

	full_text

float* %151
icall8B_
]
	full_textP
N
L%400 = tail call float @llvm.fmuladd.f32(float %398, float %399, float %397)
*float8B

	full_text


float %398
*float8B

	full_text


float %399
*float8B

	full_text


float %397
Nload8BD
B
	full_text5
3
1%401 = load float, float* %153, align 4, !tbaa !8
,float*8B

	full_text

float* %153
Nload8BD
B
	full_text5
3
1%402 = load float, float* %155, align 4, !tbaa !8
,float*8B

	full_text

float* %155
icall8B_
]
	full_textP
N
L%403 = tail call float @llvm.fmuladd.f32(float %401, float %402, float %400)
*float8B

	full_text


float %401
*float8B

	full_text


float %402
*float8B

	full_text


float %400
Nload8BD
B
	full_text5
3
1%404 = load float, float* %157, align 4, !tbaa !8
,float*8B

	full_text

float* %157
Nload8BD
B
	full_text5
3
1%405 = load float, float* %159, align 4, !tbaa !8
,float*8B

	full_text

float* %159
icall8B_
]
	full_textP
N
L%406 = tail call float @llvm.fmuladd.f32(float %404, float %405, float %403)
*float8B

	full_text


float %404
*float8B

	full_text


float %405
*float8B

	full_text


float %403
Nload8BD
B
	full_text5
3
1%407 = load float, float* %161, align 4, !tbaa !8
,float*8B

	full_text

float* %161
Nload8BD
B
	full_text5
3
1%408 = load float, float* %163, align 4, !tbaa !8
,float*8B

	full_text

float* %163
icall8B_
]
	full_textP
N
L%409 = tail call float @llvm.fmuladd.f32(float %407, float %408, float %406)
*float8B

	full_text


float %407
*float8B

	full_text


float %408
*float8B

	full_text


float %406
Nload8BD
B
	full_text5
3
1%410 = load float, float* %165, align 4, !tbaa !8
,float*8B

	full_text

float* %165
Nload8BD
B
	full_text5
3
1%411 = load float, float* %167, align 4, !tbaa !8
,float*8B

	full_text

float* %167
icall8B_
]
	full_textP
N
L%412 = tail call float @llvm.fmuladd.f32(float %410, float %411, float %409)
*float8B

	full_text


float %410
*float8B

	full_text


float %411
*float8B

	full_text


float %409
Nload8BD
B
	full_text5
3
1%413 = load float, float* %169, align 4, !tbaa !8
,float*8B

	full_text

float* %169
Nload8BD
B
	full_text5
3
1%414 = load float, float* %171, align 4, !tbaa !8
,float*8B

	full_text

float* %171
icall8B_
]
	full_textP
N
L%415 = tail call float @llvm.fmuladd.f32(float %413, float %414, float %412)
*float8B

	full_text


float %413
*float8B

	full_text


float %414
*float8B

	full_text


float %412
Nload8BD
B
	full_text5
3
1%416 = load float, float* %173, align 4, !tbaa !8
,float*8B

	full_text

float* %173
Nload8BD
B
	full_text5
3
1%417 = load float, float* %175, align 4, !tbaa !8
,float*8B

	full_text

float* %175
icall8B_
]
	full_textP
N
L%418 = tail call float @llvm.fmuladd.f32(float %416, float %417, float %415)
*float8B

	full_text


float %416
*float8B

	full_text


float %417
*float8B

	full_text


float %415
Nload8BD
B
	full_text5
3
1%419 = load float, float* %177, align 4, !tbaa !8
,float*8B

	full_text

float* %177
Nload8BD
B
	full_text5
3
1%420 = load float, float* %179, align 4, !tbaa !8
,float*8B

	full_text

float* %179
icall8B_
]
	full_textP
N
L%421 = tail call float @llvm.fmuladd.f32(float %419, float %420, float %418)
*float8B

	full_text


float %419
*float8B

	full_text


float %420
*float8B

	full_text


float %418
Nload8BD
B
	full_text5
3
1%422 = load float, float* %181, align 4, !tbaa !8
,float*8B

	full_text

float* %181
Nload8BD
B
	full_text5
3
1%423 = load float, float* %183, align 4, !tbaa !8
,float*8B

	full_text

float* %183
icall8B_
]
	full_textP
N
L%424 = tail call float @llvm.fmuladd.f32(float %422, float %423, float %421)
*float8B

	full_text


float %422
*float8B

	full_text


float %423
*float8B

	full_text


float %421
Nload8BD
B
	full_text5
3
1%425 = load float, float* %185, align 4, !tbaa !8
,float*8B

	full_text

float* %185
Nload8BD
B
	full_text5
3
1%426 = load float, float* %187, align 4, !tbaa !8
,float*8B

	full_text

float* %187
icall8B_
]
	full_textP
N
L%427 = tail call float @llvm.fmuladd.f32(float %425, float %426, float %424)
*float8B

	full_text


float %425
*float8B

	full_text


float %426
*float8B

	full_text


float %424
Nload8BD
B
	full_text5
3
1%428 = load float, float* %189, align 4, !tbaa !8
,float*8B

	full_text

float* %189
Nload8BD
B
	full_text5
3
1%429 = load float, float* %191, align 4, !tbaa !8
,float*8B

	full_text

float* %191
icall8B_
]
	full_textP
N
L%430 = tail call float @llvm.fmuladd.f32(float %428, float %429, float %427)
*float8B

	full_text


float %428
*float8B

	full_text


float %429
*float8B

	full_text


float %427
Nload8BD
B
	full_text5
3
1%431 = load float, float* %193, align 4, !tbaa !8
,float*8B

	full_text

float* %193
Nload8BD
B
	full_text5
3
1%432 = load float, float* %195, align 4, !tbaa !8
,float*8B

	full_text

float* %195
icall8B_
]
	full_textP
N
L%433 = tail call float @llvm.fmuladd.f32(float %431, float %432, float %430)
*float8B

	full_text


float %431
*float8B

	full_text


float %432
*float8B

	full_text


float %430
Nload8BD
B
	full_text5
3
1%434 = load float, float* %197, align 4, !tbaa !8
,float*8B

	full_text

float* %197
Nload8BD
B
	full_text5
3
1%435 = load float, float* %199, align 4, !tbaa !8
,float*8B

	full_text

float* %199
icall8B_
]
	full_textP
N
L%436 = tail call float @llvm.fmuladd.f32(float %434, float %435, float %433)
*float8B

	full_text


float %434
*float8B

	full_text


float %435
*float8B

	full_text


float %433
Nload8BD
B
	full_text5
3
1%437 = load float, float* %201, align 4, !tbaa !8
,float*8B

	full_text

float* %201
Nload8BD
B
	full_text5
3
1%438 = load float, float* %203, align 4, !tbaa !8
,float*8B

	full_text

float* %203
icall8B_
]
	full_textP
N
L%439 = tail call float @llvm.fmuladd.f32(float %437, float %438, float %436)
*float8B

	full_text


float %437
*float8B

	full_text


float %438
*float8B

	full_text


float %436
Nload8BD
B
	full_text5
3
1%440 = load float, float* %205, align 4, !tbaa !8
,float*8B

	full_text

float* %205
Nload8BD
B
	full_text5
3
1%441 = load float, float* %207, align 4, !tbaa !8
,float*8B

	full_text

float* %207
icall8B_
]
	full_textP
N
L%442 = tail call float @llvm.fmuladd.f32(float %440, float %441, float %439)
*float8B

	full_text


float %440
*float8B

	full_text


float %441
*float8B

	full_text


float %439
Nload8BD
B
	full_text5
3
1%443 = load float, float* %209, align 4, !tbaa !8
,float*8B

	full_text

float* %209
Nload8BD
B
	full_text5
3
1%444 = load float, float* %211, align 4, !tbaa !8
,float*8B

	full_text

float* %211
icall8B_
]
	full_textP
N
L%445 = tail call float @llvm.fmuladd.f32(float %443, float %444, float %442)
*float8B

	full_text


float %443
*float8B

	full_text


float %444
*float8B

	full_text


float %442
Nload8BD
B
	full_text5
3
1%446 = load float, float* %213, align 4, !tbaa !8
,float*8B

	full_text

float* %213
Nload8BD
B
	full_text5
3
1%447 = load float, float* %215, align 4, !tbaa !8
,float*8B

	full_text

float* %215
icall8B_
]
	full_textP
N
L%448 = tail call float @llvm.fmuladd.f32(float %446, float %447, float %445)
*float8B

	full_text


float %446
*float8B

	full_text


float %447
*float8B

	full_text


float %445
Nload8BD
B
	full_text5
3
1%449 = load float, float* %217, align 4, !tbaa !8
,float*8B

	full_text

float* %217
Nload8BD
B
	full_text5
3
1%450 = load float, float* %219, align 4, !tbaa !8
,float*8B

	full_text

float* %219
icall8B_
]
	full_textP
N
L%451 = tail call float @llvm.fmuladd.f32(float %449, float %450, float %448)
*float8B

	full_text


float %449
*float8B

	full_text


float %450
*float8B

	full_text


float %448
Nload8BD
B
	full_text5
3
1%452 = load float, float* %221, align 4, !tbaa !8
,float*8B

	full_text

float* %221
Nload8BD
B
	full_text5
3
1%453 = load float, float* %223, align 4, !tbaa !8
,float*8B

	full_text

float* %223
icall8B_
]
	full_textP
N
L%454 = tail call float @llvm.fmuladd.f32(float %452, float %453, float %451)
*float8B

	full_text


float %452
*float8B

	full_text


float %453
*float8B

	full_text


float %451
Nload8BD
B
	full_text5
3
1%455 = load float, float* %225, align 4, !tbaa !8
,float*8B

	full_text

float* %225
Nload8BD
B
	full_text5
3
1%456 = load float, float* %227, align 4, !tbaa !8
,float*8B

	full_text

float* %227
icall8B_
]
	full_textP
N
L%457 = tail call float @llvm.fmuladd.f32(float %455, float %456, float %454)
*float8B

	full_text


float %455
*float8B

	full_text


float %456
*float8B

	full_text


float %454
Nload8BD
B
	full_text5
3
1%458 = load float, float* %229, align 4, !tbaa !8
,float*8B

	full_text

float* %229
Nload8BD
B
	full_text5
3
1%459 = load float, float* %231, align 4, !tbaa !8
,float*8B

	full_text

float* %231
icall8B_
]
	full_textP
N
L%460 = tail call float @llvm.fmuladd.f32(float %458, float %459, float %457)
*float8B

	full_text


float %458
*float8B

	full_text


float %459
*float8B

	full_text


float %457
Nload8BD
B
	full_text5
3
1%461 = load float, float* %233, align 4, !tbaa !8
,float*8B

	full_text

float* %233
Nload8BD
B
	full_text5
3
1%462 = load float, float* %235, align 4, !tbaa !8
,float*8B

	full_text

float* %235
icall8B_
]
	full_textP
N
L%463 = tail call float @llvm.fmuladd.f32(float %461, float %462, float %460)
*float8B

	full_text


float %461
*float8B

	full_text


float %462
*float8B

	full_text


float %460
Nload8BD
B
	full_text5
3
1%464 = load float, float* %237, align 4, !tbaa !8
,float*8B

	full_text

float* %237
Nload8BD
B
	full_text5
3
1%465 = load float, float* %239, align 4, !tbaa !8
,float*8B

	full_text

float* %239
icall8B_
]
	full_textP
N
L%466 = tail call float @llvm.fmuladd.f32(float %464, float %465, float %463)
*float8B

	full_text


float %464
*float8B

	full_text


float %465
*float8B

	full_text


float %463
Nload8BD
B
	full_text5
3
1%467 = load float, float* %241, align 4, !tbaa !8
,float*8B

	full_text

float* %241
Nload8BD
B
	full_text5
3
1%468 = load float, float* %243, align 4, !tbaa !8
,float*8B

	full_text

float* %243
icall8B_
]
	full_textP
N
L%469 = tail call float @llvm.fmuladd.f32(float %467, float %468, float %466)
*float8B

	full_text


float %467
*float8B

	full_text


float %468
*float8B

	full_text


float %466
Nload8BD
B
	full_text5
3
1%470 = load float, float* %245, align 4, !tbaa !8
,float*8B

	full_text

float* %245
Nload8BD
B
	full_text5
3
1%471 = load float, float* %247, align 4, !tbaa !8
,float*8B

	full_text

float* %247
icall8B_
]
	full_textP
N
L%472 = tail call float @llvm.fmuladd.f32(float %470, float %471, float %469)
*float8B

	full_text


float %470
*float8B

	full_text


float %471
*float8B

	full_text


float %469
Nload8BD
B
	full_text5
3
1%473 = load float, float* %249, align 4, !tbaa !8
,float*8B

	full_text

float* %249
Nload8BD
B
	full_text5
3
1%474 = load float, float* %251, align 4, !tbaa !8
,float*8B

	full_text

float* %251
icall8B_
]
	full_textP
N
L%475 = tail call float @llvm.fmuladd.f32(float %473, float %474, float %472)
*float8B

	full_text


float %473
*float8B

	full_text


float %474
*float8B

	full_text


float %472
Nload8BD
B
	full_text5
3
1%476 = load float, float* %253, align 4, !tbaa !8
,float*8B

	full_text

float* %253
Nload8BD
B
	full_text5
3
1%477 = load float, float* %255, align 4, !tbaa !8
,float*8B

	full_text

float* %255
icall8B_
]
	full_textP
N
L%478 = tail call float @llvm.fmuladd.f32(float %476, float %477, float %475)
*float8B

	full_text


float %476
*float8B

	full_text


float %477
*float8B

	full_text


float %475
Nload8BD
B
	full_text5
3
1%479 = load float, float* %257, align 4, !tbaa !8
,float*8B

	full_text

float* %257
Nload8BD
B
	full_text5
3
1%480 = load float, float* %259, align 4, !tbaa !8
,float*8B

	full_text

float* %259
icall8B_
]
	full_textP
N
L%481 = tail call float @llvm.fmuladd.f32(float %479, float %480, float %478)
*float8B

	full_text


float %479
*float8B

	full_text


float %480
*float8B

	full_text


float %478
Nload8BD
B
	full_text5
3
1%482 = load float, float* %261, align 4, !tbaa !8
,float*8B

	full_text

float* %261
Nload8BD
B
	full_text5
3
1%483 = load float, float* %263, align 4, !tbaa !8
,float*8B

	full_text

float* %263
icall8B_
]
	full_textP
N
L%484 = tail call float @llvm.fmuladd.f32(float %482, float %483, float %481)
*float8B

	full_text


float %482
*float8B

	full_text


float %483
*float8B

	full_text


float %481
Nload8BD
B
	full_text5
3
1%485 = load float, float* %265, align 4, !tbaa !8
,float*8B

	full_text

float* %265
Nload8BD
B
	full_text5
3
1%486 = load float, float* %267, align 4, !tbaa !8
,float*8B

	full_text

float* %267
icall8B_
]
	full_textP
N
L%487 = tail call float @llvm.fmuladd.f32(float %485, float %486, float %484)
*float8B

	full_text


float %485
*float8B

	full_text


float %486
*float8B

	full_text


float %484
Nload8BD
B
	full_text5
3
1%488 = load float, float* %269, align 4, !tbaa !8
,float*8B

	full_text

float* %269
Nload8BD
B
	full_text5
3
1%489 = load float, float* %271, align 4, !tbaa !8
,float*8B

	full_text

float* %271
icall8B_
]
	full_textP
N
L%490 = tail call float @llvm.fmuladd.f32(float %488, float %489, float %487)
*float8B

	full_text


float %488
*float8B

	full_text


float %489
*float8B

	full_text


float %487
Nload8BD
B
	full_text5
3
1%491 = load float, float* %273, align 4, !tbaa !8
,float*8B

	full_text

float* %273
Nload8BD
B
	full_text5
3
1%492 = load float, float* %275, align 4, !tbaa !8
,float*8B

	full_text

float* %275
icall8B_
]
	full_textP
N
L%493 = tail call float @llvm.fmuladd.f32(float %491, float %492, float %490)
*float8B

	full_text


float %491
*float8B

	full_text


float %492
*float8B

	full_text


float %490
Nload8BD
B
	full_text5
3
1%494 = load float, float* %277, align 4, !tbaa !8
,float*8B

	full_text

float* %277
Nload8BD
B
	full_text5
3
1%495 = load float, float* %279, align 4, !tbaa !8
,float*8B

	full_text

float* %279
icall8B_
]
	full_textP
N
L%496 = tail call float @llvm.fmuladd.f32(float %494, float %495, float %493)
*float8B

	full_text


float %494
*float8B

	full_text


float %495
*float8B

	full_text


float %493
Nload8BD
B
	full_text5
3
1%497 = load float, float* %281, align 4, !tbaa !8
,float*8B

	full_text

float* %281
Nload8BD
B
	full_text5
3
1%498 = load float, float* %283, align 4, !tbaa !8
,float*8B

	full_text

float* %283
icall8B_
]
	full_textP
N
L%499 = tail call float @llvm.fmuladd.f32(float %497, float %498, float %496)
*float8B

	full_text


float %497
*float8B

	full_text


float %498
*float8B

	full_text


float %496
Nload8BD
B
	full_text5
3
1%500 = load float, float* %285, align 4, !tbaa !8
,float*8B

	full_text

float* %285
Nload8BD
B
	full_text5
3
1%501 = load float, float* %287, align 4, !tbaa !8
,float*8B

	full_text

float* %287
icall8B_
]
	full_textP
N
L%502 = tail call float @llvm.fmuladd.f32(float %500, float %501, float %499)
*float8B

	full_text


float %500
*float8B

	full_text


float %501
*float8B

	full_text


float %499
Nload8BD
B
	full_text5
3
1%503 = load float, float* %289, align 4, !tbaa !8
,float*8B

	full_text

float* %289
Nload8BD
B
	full_text5
3
1%504 = load float, float* %291, align 4, !tbaa !8
,float*8B

	full_text

float* %291
icall8B_
]
	full_textP
N
L%505 = tail call float @llvm.fmuladd.f32(float %503, float %504, float %502)
*float8B

	full_text


float %503
*float8B

	full_text


float %504
*float8B

	full_text


float %502
Nload8BD
B
	full_text5
3
1%506 = load float, float* %293, align 4, !tbaa !8
,float*8B

	full_text

float* %293
Nload8BD
B
	full_text5
3
1%507 = load float, float* %295, align 4, !tbaa !8
,float*8B

	full_text

float* %295
icall8B_
]
	full_textP
N
L%508 = tail call float @llvm.fmuladd.f32(float %506, float %507, float %505)
*float8B

	full_text


float %506
*float8B

	full_text


float %507
*float8B

	full_text


float %505
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
3add8B*
(
	full_text

%509 = add i64 %302, 64
&i648B

	full_text


i64 %302
4add8B+
)
	full_text

%510 = add i64 %303, %40
&i648B

	full_text


i64 %303
%i648B

	full_text
	
i64 %40
:icmp8B0
.
	full_text!

%511 = icmp slt i64 %509, %42
&i648B

	full_text


i64 %509
%i648B

	full_text
	
i64 %42
=br8B5
3
	full_text&
$
"br i1 %511, label %301, label %296
$i18B

	full_text
	
i1 %511
Qcall8BG
E
	full_text8
6
4%513 = tail call i64 @_Z15get_global_sizej(i32 0) #4
5mul8B,
*
	full_text

%514 = mul i64 %513, %298
&i648B

	full_text


i64 %513
&i648B

	full_text


i64 %298
Ocall8BE
C
	full_text6
4
2%515 = tail call i64 @_Z13get_global_idj(i32 0) #4
5add8B,
*
	full_text

%516 = add i64 %514, %515
&i648B

	full_text


i64 %514
&i648B

	full_text


i64 %515
^getelementptr8BK
I
	full_text<
:
8%517 = getelementptr inbounds float, float* %0, i64 %516
&i648B

	full_text


i64 %516
Nstore8BC
A
	full_text4
2
0store float %297, float* %517, align 4, !tbaa !8
*float8B

	full_text


float %297
,float*8B

	full_text

float* %517
(br8B 

	full_text

br label %518
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %6
*float*8B

	full_text

	float* %1
*float*8B

	full_text

	float* %2
$i328B

	full_text


i32 %7
$i328B

	full_text


i32 %5
*float*8B

	full_text

	float* %4
*float*8B

	full_text

	float* %3
*float*8B

	full_text

	float* %0
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
&i648B

	full_text


i64 2432
%i648B

	full_text
	
i64 320
&i648B

	full_text


i64 1472
%i648B

	full_text
	
i64 512
$i648B

	full_text


i64 11
%i648B

	full_text
	
i64 640
&i648B

	full_text


i64 1152
$i648B

	full_text


i64 25
$i648B

	full_text


i64 35
$i648B

	full_text


i64 36
&i648B

	full_text


i64 2112
&i648B

	full_text


i64 1536
$i648B

	full_text


i64 61
&i648B

	full_text


i64 2048
#i648B

	full_text	

i64 2
&i648B

	full_text


i64 2304
%i648B

	full_text
	
i64 192
&i648B

	full_text


i64 2560
%i648B

	full_text
	
i64 832
%i648B

	full_text
	
i64 576
%i648B

	full_text
	
i64 960
%i648B

	full_text
	
i64 128
$i648B

	full_text


i64 23
&i648B

	full_text


i64 1280
&i648B

	full_text


i64 2240
&i648B

	full_text


i64 3712
&i648B

	full_text


i64 2496
&i648B

	full_text


i64 3328
$i648B

	full_text


i64 12
&i648B

	full_text


i64 1728
&i648B

	full_text


i64 2752
&i648B

	full_text


i64 3840
&i648B

	full_text


i64 3904
&i648B

	full_text


i64 4032
&i648B

	full_text


i64 3584
$i648B

	full_text


i64 47
#i328B

	full_text	

i32 1
$i648B

	full_text


i64 42
&i648B

	full_text


i64 2944
$i648B

	full_text


i64 41
%i648B

	full_text
	
i64 256
#i648B

	full_text	

i64 7
$i648B

	full_text


i64 60
$i648B

	full_text


i64 48
&i648B

	full_text


i64 1856
$i648B

	full_text


i64 54
%i648B

	full_text
	
i64 896
$i648B

	full_text


i64 58
&i648B

	full_text


i64 1024
#i648B

	full_text	

i64 4
$i648B

	full_text


i64 51
$i648B

	full_text


i64 24
&i648B

	full_text


i64 1216
$i648B

	full_text


i64 62
$i648B

	full_text


i64 19
$i648B

	full_text


i64 38
&i648B

	full_text


i64 1984
&i648B

	full_text


i64 3520
#i328B

	full_text	

i32 6
$i648B

	full_text


i64 37
$i648B

	full_text


i64 40
$i648B

	full_text


i64 10
&i648B

	full_text


i64 3264
#i648B

	full_text	

i64 6
&i648B

	full_text


i64 1664
#i648B

	full_text	

i64 5
$i648B

	full_text


i64 64
$i648B

	full_text


i64 34
$i648B

	full_text


i64 55
$i648B

	full_text


i64 29
$i648B

	full_text


i64 32
$i648B

	full_text


i64 63
#i648B

	full_text	

i64 3
$i648B

	full_text


i64 13
$i648B

	full_text


i64 21
$i648B

	full_text


i64 50
&i648B

	full_text


i64 3136
&i648B

	full_text


i64 2176
#i648B

	full_text	

i64 8
$i648B

	full_text


i64 57
&i648B

	full_text


i64 3776
2float8B%
#
	full_text

float 0.000000e+00
$i648B

	full_text


i64 30
$i648B

	full_text


i64 43
&i648B

	full_text


i64 3968
$i648B

	full_text


i64 56
#i648B

	full_text	

i64 1
%i648B

	full_text
	
i64 704
%i648B

	full_text
	
i64 384
$i648B

	full_text


i64 27
$i648B

	full_text


i64 31
&i648B

	full_text


i64 2368
$i648B

	full_text


i64 28
$i648B

	full_text


i64 44
$i648B

	full_text


i64 33
$i648B

	full_text


i64 59
$i648B

	full_text


i64 14
#i328B

	full_text	

i32 0
$i648B

	full_text


i64 18
&i648B

	full_text


i64 1920
&i648B

	full_text


i64 3648
$i648B

	full_text


i64 17
&i648B

	full_text


i64 2880
&i648B

	full_text


i64 1344
&i648B

	full_text


i64 1792
$i648B

	full_text


i64 53
&i648B

	full_text


i64 2688
&i648B

	full_text


i64 3456
&i648B

	full_text


i64 1088
&i648B

	full_text


i64 2624
%i648B

	full_text
	
i64 448
$i648B

	full_text


i64 45
#i648B

	full_text	

i64 9
$i648B

	full_text


i64 22
$i648B

	full_text


i64 20
%i648B

	full_text
	
i64 768
$i648B

	full_text


i64 15
&i648B

	full_text


i64 2816
&i648B

	full_text


i64 3072
$i648B

	full_text


i64 52
&i648B

	full_text


i64 3200
$i648B

	full_text


i64 49
$i648B

	full_text


i64 39
&i648B

	full_text


i64 3008
&i648B

	full_text


i64 3392
$i648B

	full_text


i64 16
&i648B

	full_text


i64 1408
$i648B

	full_text


i64 26
$i648B

	full_text


i64 46
&i648B

	full_text


i64 1600       	  

                      !  "    #$ ## %& %% '( '' )* )) +, +- ++ ./ .. 01 00 23 22 45 44 67 66 89 88 :; :: <= << >? >> @A @@ BC BB DE DD FG FF HI HH JK JJ LM LL NO NN PQ PP RS RR TU TT VW VV XY XX Z[ ZZ \] \\ ^_ ^^ `a `` bc bb de dd fg ff hi hh jk jj lm ll no nn pq pp rs rr tu tt vw vv xy xx z{ zz |} || ~ ~~ ÄÅ ÄÄ Ç
É ÇÇ ÑÖ ÑÑ Ü
á ÜÜ àâ àà ä
ã ää åç åå é
è éé êë êê í
ì íí îï îî ñ
ó ññ òô òò ö
õ öö úù úú û
ü ûû †° †† ¢
£ ¢¢ §• §§ ¶
ß ¶¶ ®© ®® ™
´ ™™ ¨≠ ¨¨ Æ
Ø ÆÆ ∞± ∞∞ ≤
≥ ≤≤ ¥µ ¥¥ ∂
∑ ∂∂ ∏π ∏∏ ∫
ª ∫∫ ºΩ ºº æ
ø ææ ¿¡ ¿¿ ¬
√ ¬¬ ƒ≈ ƒƒ ∆
« ∆∆ »… »»  
À    ÃÕ ÃÃ Œ
œ ŒŒ –— –– “
” ““ ‘’ ‘‘ ÷
◊ ÷÷ ÿŸ ÿÿ ⁄
€ ⁄⁄ ‹› ‹‹ ﬁ
ﬂ ﬁﬁ ‡· ‡‡ ‚
„ ‚‚ ‰Â ‰‰ Ê
Á ÊÊ ËÈ ËË Í
Î ÍÍ ÏÌ ÏÏ Ó
Ô ÓÓ Ò  Ú
Û ÚÚ Ùı ÙÙ ˆ
˜ ˆˆ ¯˘ ¯¯ ˙
˚ ˙˙ ¸˝ ¸¸ ˛
ˇ ˛˛ ÄÅ ÄÄ Ç
É ÇÇ ÑÖ ÑÑ Ü
á ÜÜ àâ àà ä
ã ää åç åå é
è éé êë êê í
ì íí îï îî ñ
ó ññ òô òò ö
õ öö úù úú û
ü ûû †° †† ¢
£ ¢¢ §• §§ ¶
ß ¶¶ ®© ®® ™
´ ™™ ¨≠ ¨¨ Æ
Ø ÆÆ ∞± ∞∞ ≤
≥ ≤≤ ¥µ ¥¥ ∂
∑ ∂∂ ∏π ∏∏ ∫
ª ∫∫ ºΩ ºº æ
ø ææ ¿¡ ¿¿ ¬
√ ¬¬ ƒ≈ ƒƒ ∆
« ∆∆ »… »»  
À    ÃÕ ÃÃ Œ
œ ŒŒ –— –– “
” ““ ‘’ ‘‘ ÷
◊ ÷÷ ÿŸ ÿÿ ⁄
€ ⁄⁄ ‹› ‹‹ ﬁ
ﬂ ﬁﬁ ‡· ‡‡ ‚
„ ‚‚ ‰Â ‰‰ Ê
Á ÊÊ ËÈ ËË Í
Î ÍÍ ÏÌ ÏÏ Ó
Ô ÓÓ Ò  Ú
Û ÚÚ Ùı ÙÙ ˆ
˜ ˆˆ ¯˘ ¯¯ ˙
˚ ˙˙ ¸˝ ¸¸ ˛
ˇ ˛˛ ÄÅ ÄÄ Ç
É ÇÇ ÑÖ ÑÑ Ü
á ÜÜ àâ àà ä
ã ää åç åå é
è éé êë êê í
ì íí îï îî ñ
ó ññ òô òò ö
õ öö úù úú û
ü ûû †° †† ¢
£ ¢¢ §• §§ ¶
ß ¶¶ ®© ®® ™
´ ™™ ¨≠ ¨¨ Æ
Ø ÆÆ ∞± ∞∞ ≤
≥ ≤≤ ¥µ ¥¥ ∂
∑ ∂∂ ∏π ∏∏ ∫
ª ∫∫ ºΩ ºº æ
ø ææ ¿¡ ¿¿ ¬
√ ¬¬ ƒ≈ ƒƒ ∆
« ∆∆ »… »»  
À    ÃÕ ÃÃ Œ
œ ŒŒ –— –– “
” ““ ‘’ ‘‘ ÷
◊ ÷÷ ÿŸ ÿÿ ⁄
€ ⁄⁄ ‹› ‹‹ ﬁ
ﬂ ﬁﬁ ‡· ‡‡ ‚
„ ‚‚ ‰Â ‰‰ Ê
Á ÊÊ ËÈ ËË Í
Î ÍÍ ÏÌ ÏÏ Ó
Ô ÓÓ Ò  Ú
Û ÚÚ Ùı ÙÙ ˆ
˜ ˆˆ ¯˘ ¯¯ ˙
˚ ˙˙ ¸˝ ¸¸ ˛
ˇ ˛˛ ÄÅ ÄÄ Ç
É ÇÇ ÑÖ ÑÑ Ü
á ÜÜ àâ àà ä
ã ää åç åå é
è éé êë êê í
ì íí îï îî ñ
ó ññ òô òò ö
õ öö úù úú û
ü ûû †° †† ¢
£ ¢¢ §• §§ ¶
ß ¶¶ ®© ®® ™
´ ™™ ¨≠ ¨¨ Æ
Ø ÆÆ ∞± ∞∞ ≤
≥ ≤≤ ¥µ ¥¥ ∂
∑ ∂∂ ∏π ∏∏ ∫
ª ∫∫ º
æ ΩΩ øø ¿¿ ¡¬ ¡
√ ¡¡ ƒ≈ ƒ« ∆
» ∆∆ …  …
À …… Ã
Õ ÃÃ Œœ ŒŒ –— –
“ –– ”‘ ”” ’
÷ ’’ ◊ÿ ◊◊ Ÿ⁄ ŸŸ €‹ €
› €€ ﬁﬂ ﬁﬁ ‡· ‡
‚ ‡‡ „‰ „„ Â
Ê ÂÂ ÁË ÁÁ ÈÍ ÈÈ ÎÏ Î
Ì ÎÎ ÓÓ Ô ÔÔ ÒÚ ÒÒ ÛÙ Û
ı Û
ˆ ÛÛ ˜¯ ˜˜ ˘˙ ˘˘ ˚¸ ˚
˝ ˚
˛ ˚˚ ˇÄ ˇˇ ÅÇ ÅÅ ÉÑ É
Ö É
Ü ÉÉ áà áá âä ââ ãå ã
ç ã
é ãã èê èè ëí ëë ìî ì
ï ì
ñ ìì óò óó ôö ôô õú õ
ù õ
û õõ ü† üü °¢ °° £§ £
• £
¶ ££ ß® ßß ©™ ©© ´¨ ´
≠ ´
Æ ´´ Ø∞ ØØ ±≤ ±± ≥¥ ≥
µ ≥
∂ ≥≥ ∑∏ ∑∑ π∫ ππ ªº ª
Ω ª
æ ªª ø¿ øø ¡¬ ¡¡ √ƒ √
≈ √
∆ √√ «» «« …  …… ÀÃ À
Õ À
Œ ÀÀ œ– œœ —“ —— ”‘ ”
’ ”
÷ ”” ◊ÿ ◊◊ Ÿ⁄ ŸŸ €‹ €
› €
ﬁ €€ ﬂ‡ ﬂﬂ ·‚ ·· „‰ „
Â „
Ê „„ ÁË ÁÁ ÈÍ ÈÈ ÎÏ Î
Ì Î
Ó ÎÎ Ô ÔÔ ÒÚ ÒÒ ÛÙ Û
ı Û
ˆ ÛÛ ˜¯ ˜˜ ˘˙ ˘˘ ˚¸ ˚
˝ ˚
˛ ˚˚ ˇÄ ˇˇ ÅÇ ÅÅ ÉÑ É
Ö É
Ü ÉÉ áà áá âä ââ ãå ã
ç ã
é ãã èê èè ëí ëë ìî ì
ï ì
ñ ìì óò óó ôö ôô õú õ
ù õ
û õõ ü† üü °¢ °° £§ £
• £
¶ ££ ß® ßß ©™ ©© ´¨ ´
≠ ´
Æ ´´ Ø∞ ØØ ±≤ ±± ≥¥ ≥
µ ≥
∂ ≥≥ ∑∏ ∑∑ π∫ ππ ªº ª
Ω ª
æ ªª ø¿ øø ¡¬ ¡¡ √ƒ √
≈ √
∆ √√ «» «« …  …… ÀÃ À
Õ À
Œ ÀÀ œ– œœ —“ —— ”‘ ”
’ ”
÷ ”” ◊ÿ ◊◊ Ÿ⁄ ŸŸ €‹ €
› €
ﬁ €€ ﬂ‡ ﬂﬂ ·‚ ·· „‰ „
Â „
Ê „„ ÁË ÁÁ ÈÍ ÈÈ ÎÏ Î
Ì Î
Ó ÎÎ Ô ÔÔ ÒÚ ÒÒ ÛÙ Û
ı Û
ˆ ÛÛ ˜¯ ˜˜ ˘˙ ˘˘ ˚¸ ˚
˝ ˚
˛ ˚˚ ˇÄ ˇˇ ÅÇ ÅÅ ÉÑ É
Ö É
Ü ÉÉ áà áá âä ââ ãå ã
ç ã
é ãã èê èè ëí ëë ìî ì
ï ì
ñ ìì óò óó ôö ôô õú õ
ù õ
û õõ ü† üü °¢ °° £§ £
• £
¶ ££ ß® ßß ©™ ©© ´¨ ´
≠ ´
Æ ´´ Ø∞ ØØ ±≤ ±± ≥¥ ≥
µ ≥
∂ ≥≥ ∑∏ ∑∑ π∫ ππ ªº ª
Ω ª
æ ªª ø¿ øø ¡¬ ¡¡ √ƒ √
≈ √
∆ √√ «» «« …  …… ÀÃ À
Õ À
Œ ÀÀ œ– œœ —“ —— ”‘ ”
’ ”
÷ ”” ◊ÿ ◊◊ Ÿ⁄ ŸŸ €‹ €
› €
ﬁ €€ ﬂ‡ ﬂﬂ ·‚ ·· „‰ „
Â „
Ê „„ ÁË ÁÁ ÈÍ ÈÈ ÎÏ Î
Ì Î
Ó ÎÎ Ô ÔÔ ÒÚ ÒÒ ÛÙ Û
ı Û
ˆ ÛÛ ˜¯ ˜˜ ˘˙ ˘˘ ˚¸ ˚
˝ ˚
˛ ˚˚ ˇÄ ˇˇ ÅÇ ÅÅ ÉÑ É
Ö É
Ü ÉÉ áà áá âä ââ ãå ã
ç ã
é ãã èê èè ëí ëë ìî ì
ï ì
ñ ìì óò óó ôö ôô õú õ
ù õ
û õõ ü† üü °¢ °° £§ £
• £
¶ ££ ß® ßß ©™ ©© ´¨ ´
≠ ´
Æ ´´ Ø∞ ØØ ±≤ ±± ≥¥ ≥
µ ≥
∂ ≥≥ ∑∏ ∑∑ π∫ ππ ªº ª
Ω ª
æ ªª ø¿ øø ¡¬ ¡¡ √ƒ √
≈ √
∆ √√ «» «« …  …… ÀÃ À
Õ À
Œ ÀÀ œ– œœ —“ —— ”‘ ”
’ ”
÷ ”” ◊ÿ ◊◊ Ÿ⁄ ŸŸ €‹ €
› €
ﬁ €€ ﬂ‡ ﬂﬂ ·‚ ·· „‰ „
Â „
Ê „„ ÁË ÁÁ ÈÍ ÈÈ ÎÏ Î
Ì Î
Ó ÎÎ ÔÔ Ò  ÚÛ Ú
Ù ÚÚ ıˆ ı
˜ ıı ¯˘ ¯˙ ˚¸ ˚
˝ ˚˚ ˛˛ ˇÄ	 ˇ
Å	 ˇˇ Ç	
É	 Ç	Ç	 Ñ	Ö	 Ñ	
Ü	 Ñ	Ñ	 á	â	 	â	 )ä	 ’ã	 Âå	 ¿ç	 
ç	 	ç	 	ç	 é	 .é	 Bé	 Jé	 Ré	 Zé	 bé	 jé	 ré	 zé	 Çé	 äé	 íé	 öé	 ¢é	 ™é	 ≤é	 ∫é	 ¬é	  é	 “é	 ⁄é	 ‚é	 Íé	 Úé	 ˙é	 Çé	 äé	 íé	 öé	 ¢é	 ™é	 ≤é	 ∫é	 ¬é	  é	 “é	 ⁄é	 ‚é	 Íé	 Úé	 ˙é	 Çé	 äé	 íé	 öé	 ¢é	 ™é	 ≤é	 ∫é	 ¬é	  é	 “é	 ⁄é	 ‚é	 Íé	 Úé	 ˙é	 Çé	 äé	 íé	 öé	 ¢é	 ™é	 ≤é	 ∫è	 %è	 8è	 Fè	 Nè	 Vè	 ^è	 fè	 nè	 vè	 ~è	 Üè	 éè	 ñè	 ûè	 ¶è	 Æè	 ∂è	 æè	 ∆è	 Œè	 ÷è	 ﬁè	 Êè	 Óè	 ˆè	 ˛è	 Üè	 éè	 ñè	 ûè	 ¶è	 Æè	 ∂è	 æè	 ∆è	 Œè	 ÷è	 ﬁè	 Êè	 Óè	 ˆè	 ˛è	 Üè	 éè	 ñè	 ûè	 ¶è	 Æè	 ∂è	 æè	 ∆è	 Œè	 ÷è	 ﬁè	 Êè	 Óè	 ˆè	 ˛è	 Üè	 éè	 ñè	 ûè	 ¶è	 Æè	 ∂ê	 Ç	   	
           ! "  $# &% ( *) , -# /. 1 3 54 72 9 ; = ? A6 C2 ED G6 IH K2 ML O6 QP S2 UT W6 YX [2 ]\ _6 a` c2 ed g6 ih k2 ml o6 qp s2 ut w6 yx {2 }| 6 ÅÄ É2 ÖÑ á6 âà ã2 çå è6 ëê ì2 ïî ó6 ôò õ2 ùú ü6 °† £2 •§ ß6 ©® ´2 ≠¨ Ø6 ±∞ ≥2 µ¥ ∑6 π∏ ª2 Ωº ø6 ¡¿ √2 ≈ƒ «6 …» À2 ÕÃ œ6 —– ”2 ’‘ ◊6 Ÿÿ €2 ›‹ ﬂ6 ·‡ „2 Â‰ Á6 ÈË Î2 ÌÏ Ô6 Ò Û2 ıÙ ˜6 ˘¯ ˚2 ˝¸ ˇ6 ÅÄ É2 ÖÑ á6 âà ã2 çå è6 ëê ì2 ïî ó6 ôò õ2 ùú ü6 °† £2 •§ ß6 ©® ´2 ≠¨ Ø6 ±∞ ≥2 µ¥ ∑6 π∏ ª2 Ωº ø6 ¡¿ √2 ≈ƒ «6 …» À2 ÕÃ œ6 —– ”2 ’‘ ◊6 Ÿÿ €2 ›‹ ﬂ6 ·‡ „2 Â‰ Á6 ÈË Î2 ÌÏ Ô6 Ò Û2 ıÙ ˜6 ˘¯ ˚2 ˝¸ ˇ6 ÅÄ É2 ÖÑ á6 âà ã2 çå è6 ëê ì2 ïî ó6 ôò õ2 ùú ü6 °† £2 •§ ß6 ©® ´2 ≠¨ Ø6 ±∞ ≥2 µ¥ ∑6 π∏ ª2 Ωº ø6 ¡¿ √2 ≈ƒ «6 …» À2 ÕÃ œ6 —– ”2 ’‘ ◊6 Ÿÿ €2 ›‹ ﬂ6 ·‡ „2 Â‰ Á6 ÈË Î2 ÌÏ Ô6 Ò Û2 ıÙ ˜6 ˘¯ ˚2 ˝¸ ˇ6 ÅÄ É2 ÖÑ á6 âà ã2 çå è6 ëê ì2 ïî ó6 ôò õ2 ùú ü6 °† £2 •§ ß6 ©® ´2 ≠¨ Ø6 ±∞ ≥2 µ¥ ∑6 π∏ ªÎ æø ¬¿ √¡ ≈> « »:  Ú ÀÎ Õ∆ œ —Œ “– ‘” ÷’ ÿ◊ ⁄Ÿ ‹' ›… ﬂ+ ·ﬁ ‚‡ ‰„ ÊÂ ËÁ ÍÈ Ï0 Ì8 B ÚÔ ÙÒ ıÃ ˆF ¯J ˙˜ ¸˘ ˝Û ˛N ÄR Çˇ ÑÅ Ö˚ ÜV àZ äá åâ çÉ é^ êb íè îë ïã ñf òj öó úô ùì ûn †r ¢ü §° •õ ¶v ®z ™ß ¨© ≠£ Æ~ ∞Ç ≤Ø ¥± µ´ ∂Ü ∏ä ∫∑ ºπ Ω≥ æé ¿í ¬ø ƒ¡ ≈ª ∆ñ »ö  « Ã… Õ√ Œû –¢ “œ ‘— ’À ÷¶ ÿ™ ⁄◊ ‹Ÿ ›” ﬁÆ ‡≤ ‚ﬂ ‰· Â€ Ê∂ Ë∫ ÍÁ ÏÈ Ì„ Óæ ¬ ÚÔ ÙÒ ıÎ ˆ∆ ¯  ˙˜ ¸˘ ˝Û ˛Œ Ä“ Çˇ ÑÅ Ö˚ Ü÷ à⁄ äá åâ çÉ éﬁ ê‚ íè îë ïã ñÊ òÍ öó úô ùì ûÓ †Ú ¢ü §° •õ ¶ˆ ®˙ ™ß ¨© ≠£ Æ˛ ∞Ç ≤Ø ¥± µ´ ∂Ü ∏ä ∫∑ ºπ Ω≥ æé ¿í ¬ø ƒ¡ ≈ª ∆ñ »ö  « Ã… Õ√ Œû –¢ “œ ‘— ’À ÷¶ ÿ™ ⁄◊ ‹Ÿ ›” ﬁÆ ‡≤ ‚ﬂ ‰· Â€ Ê∂ Ë∫ ÍÁ ÏÈ Ì„ Óæ ¬ ÚÔ ÙÒ ıÎ ˆ∆ ¯  ˙˜ ¸˘ ˝Û ˛Œ Ä“ Çˇ ÑÅ Ö˚ Ü÷ à⁄ äá åâ çÉ éﬁ ê‚ íè îë ïã ñÊ òÍ öó úô ùì ûÓ †Ú ¢ü §° •õ ¶ˆ ®˙ ™ß ¨© ≠£ Æ˛ ∞Ç ≤Ø ¥± µ´ ∂Ü ∏ä ∫∑ ºπ Ω≥ æé ¿í ¬ø ƒ¡ ≈ª ∆ñ »ö  « Ã… Õ√ Œû –¢ “œ ‘— ’À ÷¶ ÿ™ ⁄◊ ‹Ÿ ›” ﬁÆ ‡≤ ‚ﬂ ‰· Â€ Ê∂ Ë∫ ÍÁ ÏÈ Ì„ Óæ ¬ ÚÔ ÙÒ ıÎ ˆ∆ ¯  ˙˜ ¸˘ ˝Û ˛Œ Ä“ Çˇ ÑÅ Ö˚ Ü÷ à⁄ äá åâ çÉ éﬁ ê‚ íè îë ïã ñÊ òÍ öó úô ùì ûÓ †Ú ¢ü §° •õ ¶ˆ ®˙ ™ß ¨© ≠£ Æ˛ ∞Ç ≤Ø ¥± µ´ ∂Ü ∏ä ∫∑ ºπ Ω≥ æé ¿í ¬ø ƒ¡ ≈ª ∆ñ »ö  « Ã… Õ√ Œû –¢ “œ ‘— ’À ÷¶ ÿ™ ⁄◊ ‹Ÿ ›” ﬁÆ ‡≤ ‚ﬂ ‰· Â€ Ê∂ Ë∫ ÍÁ ÏÈ Ì„ Ó∆ Ò… Û< Ù ˆ@ ˜ı ˘˙ ¸ø ˝˚ Ä	˛ Å	ˇ É	Ω Ö	Ç	 Ü	  Ωº ∆ƒ ˙ƒ à	¯ ∆¯ Ωá	 à	 à	 í	í	 ì	ì	 î	î	 ñ	ñ	 ï	ï	 ë	ë	” î	î	 ”˚ î	î	 ˚£ î	î	 £≥ î	î	 ≥ã î	î	 ãÎ î	î	 Î£ î	î	 £Û î	î	 Ûª î	î	 ª€ î	î	 €Î î	î	 Î≥ î	î	 ≥ ë	ë	 ˚ î	î	 ˚ã î	î	 ã€ î	î	 €£ î	î	 £ã î	î	 ãõ î	î	 õ√ î	î	 √À î	î	 ÀÓ ì	ì	 Óõ î	î	 õì î	î	 ì˙ ñ	ñ	 ˙≥ î	î	 ≥Û î	î	 Û√ î	î	 √É î	î	 ÉÉ î	î	 Éì î	î	 ìÉ î	î	 É˚ î	î	 ˚ì î	î	 ì£ î	î	 £À î	î	 À´ î	î	 ´€ î	î	 €” î	î	 ”˚ î	î	 ˚„ î	î	 „ø ï	ï	 øÛ î	î	 Ûª î	î	 ªì î	î	 ì´ î	î	 ´√ î	î	 √„ î	î	 „´ î	î	 ´ í	í	 ´ î	î	 ´≥ î	î	 ≥Î î	î	 ÎÛ î	î	 Û ë	ë	 ª î	î	 ªÀ î	î	 À” î	î	 ” í	í	 õ î	î	 õÉ î	î	 Éª î	î	 ªÀ î	î	 Àã î	î	 ãõ î	î	 õ” î	î	 ”√ î	î	 √Ô ì	ì	 Ô„ î	î	 „„ î	î	 „Î î	î	 Î€ î	î	 €˛ ï	ï	 ˛
ó	 	ò	 h
ô	 ¯
ö	 Ä
õ	 î
ú	 ê
ù	 –
û	 Ñ
ü	 ‘
†	 ‹
°	 »
¢	 Ä
£	 §
§	 ¿	•	 L
¶	 ‡	ß	 X
®	 Ä
©	 ®
™	 à
´	 ∏	¨	 P
≠	 Ù
Æ	 ‡
Ø	 ÿ
∞	 ê
±	 ¯
≤	 ‡
≥	 ú
¥	 ò
µ	 ò
∂	 †
∑	 ®
∏	 ∏
π	 Ä
∫	 ¥ª	 ª	 ª	 øª	 Óª	 Ô
º	 å
Ω	 ∞
æ	 Ñ	ø	 `	¿	 t
¡	 ú
¬	 º
√	 ®
ƒ	 Ï
≈	 ∞
∆	 å
«	 ¿	»	 \
…	 ‘
 	 ¸
À	 ÿ
Ã	 ¨
Õ	 ‘
Œ	 Ï
œ	 ∏
–	 ¯	—	 
	—	 	—	 	—	 
“	 ‰
”	 ¸
‘	 å
’	 ÿ	÷	 l
◊	 ê	ÿ	 d	Ÿ	 H
Ÿ	 
⁄	 Ã
€	 Ù
‹	 §	›	 4	›	 6
›	 º
ﬁ	 ¥	ﬂ	 T
‡	 §
·	 ‰
‚	 Ã
„	 »
‰	 –	Â	 |
Ê	 Ñ
Á	 òË	 ΩË	 Ã
È	 ¨
Í	 î
Î	 ∞
Ï	 ¸	Ì	 D
Ó	 ò	Ô	 p
	 î
Ò	 ¥
Ú	 Ë
Û	 ú
Ù	 ú
ı	 ƒ
ˆ	 î
˜	 ¨¯	 	¯	 ¯	 ¯	 ˙¯	 ˛
˘	 Ã
˙	 ∞
˚	 à
¸	 ƒ
˝	 ®
˛	 Ë
ˇ	 †
Ä
 ‰
Å
 ê
Ç
 
É
 »
Ñ
 à	Ö
 x
Ü
 §
á
 Ñ
à
 Ï
â
 ‹
ä
 †
ã
 ¥
å
 †
ç
 ¿
é
 ‹
è
 –
ê
 ƒ
ë
 Ù
í
 ∏
ì
 Ë
î
 º
ï
 
ñ
 å
ó
 ¨
ò
 à"
	matrixMul"
_Z12get_group_idj"
_Z12get_local_idj"
_Z7barrierj"
llvm.fmuladd.f32"
_Z13get_global_idj"
_Z15get_global_sizej*ö
!nvidia-4.2-MatrixMul-matrixMul.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Ä
 
transfer_bytes_log1p
KäAA

devmap_label
 

wgsize
Ä

wgsize_log1p
KäAA

transfer_bytes
Ä¯
