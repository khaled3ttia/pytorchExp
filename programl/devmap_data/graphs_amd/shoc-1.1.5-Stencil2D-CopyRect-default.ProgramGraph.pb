

[external]
JcallBB
@
	full_text3
1
/%9 = tail call i64 @_Z12get_group_idj(i32 0) #3
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
0%11 = tail call i64 @_Z12get_local_idj(i32 0) #3
6truncB-
+
	full_text

%12 = trunc i64 %11 to i32
#i64B

	full_text
	
i64 %11
McallBE
C
	full_text6
4
2%13 = tail call i64 @_Z14get_local_sizej(i32 0) #3
6truncB-
+
	full_text

%14 = trunc i64 %13 to i32
#i64B

	full_text
	
i64 %13
4mulB-
+
	full_text

%15 = mul nsw i32 %14, %10
#i32B

	full_text
	
i32 %14
#i32B

	full_text
	
i32 %10
4addB-
+
	full_text

%16 = add nsw i32 %15, %12
#i32B

	full_text
	
i32 %15
#i32B

	full_text
	
i32 %12
5icmpB-
+
	full_text

%17 = icmp slt i32 %16, %7
#i32B

	full_text
	
i32 %16
3icmpB+
)
	full_text

%18 = icmp sgt i32 %6, 0
/andB(
&
	full_text

%19 = and i1 %17, %18
!i1B

	full_text


i1 %17
!i1B

	full_text


i1 %18
8brB2
0
	full_text#
!
br i1 %19, label %20, label %38
!i1B

	full_text


i1 %19
5sext8B+
)
	full_text

%21 = sext i32 %4 to i64
\getelementptr8BI
G
	full_text:
8
6%22 = getelementptr inbounds float, float* %3, i64 %21
%i648B

	full_text
	
i64 %21
5sext8B+
)
	full_text

%23 = sext i32 %1 to i64
\getelementptr8BI
G
	full_text:
8
6%24 = getelementptr inbounds float, float* %0, i64 %23
%i648B

	full_text
	
i64 %23
'br8B

	full_text

br label %25
Bphi8B9
7
	full_text*
(
&%26 = phi i32 [ 0, %20 ], [ %36, %25 ]
%i328B

	full_text
	
i32 %36
Xcall8BN
L
	full_text?
=
;%27 = tail call i32 @ToFlatIdx(i32 %16, i32 %26, i32 %5) #4
%i328B

	full_text
	
i32 %16
%i328B

	full_text
	
i32 %26
6sext8B,
*
	full_text

%28 = sext i32 %27 to i64
%i328B

	full_text
	
i32 %27
]getelementptr8BJ
H
	full_text;
9
7%29 = getelementptr inbounds float, float* %22, i64 %28
+float*8B

	full_text


float* %22
%i648B

	full_text
	
i64 %28
@bitcast8B3
1
	full_text$
"
 %30 = bitcast float* %29 to i32*
+float*8B

	full_text


float* %29
Hload8B>
<
	full_text/
-
+%31 = load i32, i32* %30, align 4, !tbaa !8
'i32*8B

	full_text


i32* %30
Xcall8BN
L
	full_text?
=
;%32 = tail call i32 @ToFlatIdx(i32 %16, i32 %26, i32 %2) #4
%i328B

	full_text
	
i32 %16
%i328B

	full_text
	
i32 %26
6sext8B,
*
	full_text

%33 = sext i32 %32 to i64
%i328B

	full_text
	
i32 %32
]getelementptr8BJ
H
	full_text;
9
7%34 = getelementptr inbounds float, float* %24, i64 %33
+float*8B

	full_text


float* %24
%i648B

	full_text
	
i64 %33
@bitcast8B3
1
	full_text$
"
 %35 = bitcast float* %34 to i32*
+float*8B

	full_text


float* %34
Hstore8B=
;
	full_text.
,
*store i32 %31, i32* %35, align 4, !tbaa !8
%i328B

	full_text
	
i32 %31
'i32*8B

	full_text


i32* %35
8add8B/
-
	full_text 

%36 = add nuw nsw i32 %26, 1
%i328B

	full_text
	
i32 %26
6icmp8B,
*
	full_text

%37 = icmp eq i32 %36, %6
%i328B

	full_text
	
i32 %36
:br8B2
0
	full_text#
!
br i1 %37, label %38, label %25
#i18B

	full_text


i1 %37
$ret8B

	full_text


ret void
*float*8B

	full_text

	float* %0
$i328B

	full_text


i32 %6
$i328B

	full_text


i32 %1
*float*8B

	full_text

	float* %3
$i328B

	full_text


i32 %2
$i328B

	full_text


i32 %7
$i328B

	full_text


i32 %5
$i328B
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
#i328B

	full_text	

i32 0
#i328B

	full_text	

i32 1       	  
 
 

                    !" !# !! $% $$ &' &( && )* )) +, ++ -. -/ -- 01 00 23 24 22 56 55 78 79 77 :; :: <= << >? >A B B <C D E -F G !H    	  
        :   " #! % '$ (& *) , . /- 1 30 42 6+ 85 9 ;: =< ?  @ > @>  @ JJ LL KK II! LL ! KK - LL - II  JJ M M M M M N :"

CopyRect"
_Z12get_group_idj"
_Z12get_local_idj"
_Z14get_local_sizej"
	ToFlatIdx*?
 shoc-1.1.5-Stencil2D-CopyRect.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282?

wgsize_log1p
???A

devmap_label


wgsize
 

transfer_bytes
???"
 
transfer_bytes_log1p
???A