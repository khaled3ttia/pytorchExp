

[external]
JcallBB
@
	full_text3
1
/%4 = tail call i64 @_Z12get_local_idj(i32 0) #3
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
KcallBC
A
	full_text4
2
0%7 = tail call i64 @_Z13get_global_idj(i32 0) #3
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
LcallBD
B
	full_text5
3
1%9 = tail call i64 @_Z14get_local_sizej(i32 0) #3
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
-shlB&
$
	full_text

%11 = shl i32 %8, 1
"i32B

	full_text


i32 %8
4zextB,
*
	full_text

%12 = zext i32 %11 to i64
#i32B

	full_text
	
i32 %11
bgetelementptrBQ
O
	full_textB
@
>%13 = getelementptr inbounds <4 x i32>, <4 x i32>* %0, i64 %12
#i64B

	full_text
	
i64 %12
SloadBK
I
	full_text<
:
8%14 = load <4 x i32>, <4 x i32>* %13, align 16, !tbaa !9
1
<4 x i32>*B!

	full_text

<4 x i32>* %13
,orB&
$
	full_text

%15 = or i32 %11, 1
#i32B

	full_text
	
i32 %11
4zextB,
*
	full_text

%16 = zext i32 %15 to i64
#i32B

	full_text
	
i32 %15
bgetelementptrBQ
O
	full_textB
@
>%17 = getelementptr inbounds <4 x i32>, <4 x i32>* %0, i64 %16
#i64B

	full_text
	
i64 %16
SloadBK
I
	full_text<
:
8%18 = load <4 x i32>, <4 x i32>* %17, align 16, !tbaa !9
1
<4 x i32>*B!

	full_text

<4 x i32>* %17
6addB/
-
	full_text 

%19 = add <4 x i32> %18, %14
/	<4 x i32>B 

	full_text

<4 x i32> %18
/	<4 x i32>B 

	full_text

<4 x i32> %14
6andB/
-
	full_text 

%20 = and i64 %4, 4294967295
"i64B

	full_text


i64 %4
bgetelementptrBQ
O
	full_textB
@
>%21 = getelementptr inbounds <4 x i32>, <4 x i32>* %2, i64 %20
#i64B

	full_text
	
i64 %20
SstoreBJ
H
	full_text;
9
7store <4 x i32> %19, <4 x i32>* %21, align 16, !tbaa !9
/	<4 x i32>B 

	full_text

<4 x i32> %19
1
<4 x i32>*B!

	full_text

<4 x i32>* %21
@callB8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
0lshrB(
&
	full_text

%22 = lshr i32 %10, 1
#i32B

	full_text
	
i32 %10
3icmpB+
)
	full_text

%23 = icmp eq i32 %22, 0
#i32B

	full_text
	
i32 %22
8brB2
0
	full_text#
!
br i1 %23, label %25, label %24
!i1B

	full_text


i1 %23
'br8B

	full_text

br label %27
4icmp8B*
(
	full_text

%26 = icmp eq i32 %5, 0
$i328B

	full_text


i32 %5
:br8B2
0
	full_text#
!
br i1 %26, label %40, label %44
#i18B

	full_text


i1 %26
Dphi8B;
9
	full_text,
*
(%28 = phi i32 [ %38, %37 ], [ %22, %24 ]
%i328B

	full_text
	
i32 %38
%i328B

	full_text
	
i32 %22
7icmp8B-
+
	full_text

%29 = icmp ugt i32 %28, %5
%i328B

	full_text
	
i32 %28
$i328B

	full_text


i32 %5
:br8B2
0
	full_text#
!
br i1 %29, label %30, label %37
#i18B

	full_text


i1 %29
1add8B(
&
	full_text

%31 = add i32 %28, %5
%i328B

	full_text
	
i32 %28
$i328B

	full_text


i32 %5
6zext8B,
*
	full_text

%32 = zext i32 %31 to i64
%i328B

	full_text
	
i32 %31
dgetelementptr8BQ
O
	full_textB
@
>%33 = getelementptr inbounds <4 x i32>, <4 x i32>* %2, i64 %32
%i648B

	full_text
	
i64 %32
Uload8BK
I
	full_text<
:
8%34 = load <4 x i32>, <4 x i32>* %33, align 16, !tbaa !9
3
<4 x i32>*8B!

	full_text

<4 x i32>* %33
Uload8BK
I
	full_text<
:
8%35 = load <4 x i32>, <4 x i32>* %21, align 16, !tbaa !9
3
<4 x i32>*8B!

	full_text

<4 x i32>* %21
8add8B/
-
	full_text 

%36 = add <4 x i32> %35, %34
1	<4 x i32>8B 

	full_text

<4 x i32> %35
1	<4 x i32>8B 

	full_text

<4 x i32> %34
Ustore8BJ
H
	full_text;
9
7store <4 x i32> %36, <4 x i32>* %21, align 16, !tbaa !9
1	<4 x i32>8B 

	full_text

<4 x i32> %36
3
<4 x i32>*8B!

	full_text

<4 x i32>* %21
'br8B

	full_text

br label %37
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
2lshr8B(
&
	full_text

%38 = lshr i32 %28, 1
%i328B

	full_text
	
i32 %28
5icmp8B+
)
	full_text

%39 = icmp eq i32 %38, 0
%i328B

	full_text
	
i32 %38
:br8B2
0
	full_text#
!
br i1 %39, label %25, label %27
#i18B

	full_text


i1 %39
Tload8BJ
H
	full_text;
9
7%41 = load <4 x i32>, <4 x i32>* %2, align 16, !tbaa !9
8and8B/
-
	full_text 

%42 = and i64 %6, 4294967295
$i648B

	full_text


i64 %6
dgetelementptr8BQ
O
	full_textB
@
>%43 = getelementptr inbounds <4 x i32>, <4 x i32>* %1, i64 %42
%i648B

	full_text
	
i64 %42
Ustore8BJ
H
	full_text;
9
7store <4 x i32> %41, <4 x i32>* %43, align 16, !tbaa !9
1	<4 x i32>8B 

	full_text

<4 x i32> %41
3
<4 x i32>*8B!

	full_text

<4 x i32>* %43
'br8B

	full_text

br label %44
$ret8B

	full_text


ret void
2
<4 x i32>*8B 

	full_text

<4 x i32>* %2
2
<4 x i32>*8B 

	full_text

<4 x i32>* %0
2
<4 x i32>*8B 

	full_text

<4 x i32>* %1
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
#i328B

	full_text	

i32 0
,i648B!

	full_text

i64 4294967295
#i328B

	full_text	

i32 1        	
 		                       !    "# "$ "" %% &' && () (( *+ *. -- /0 /2 13 11 45 46 44 78 7: 9; 99 <= << >? >> @A @@ BC BB DE DF DD GH GI GG JK LM LL NO NN PQ PR ST SS UV UU WX WY WW Z\  \ >\ R] ] ^ U   
            ! #  $	 '& )( + .- 0L 2& 31 5 64 81 : ;9 =< ?> A  CB E@ FD H  I1 ML ON Q TS VR XU Y* -* ,/ R/ [, 1Z [7 97 KJ KP -P 1 bb cc `` aa __ [ bb  __  `` % cc %K cc K aa d d d d d (d -d Ne e Sf f f %f &f Kf L"
reduce"
_Z12get_local_idj"
_Z12get_group_idj"
_Z13get_global_idj"
_Z14get_local_sizej"
_Z7barrierj*?
Reduction_Kernels.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02
 
transfer_bytes_log1p
15A

transfer_bytes
?@

devmap_label
 

wgsize_log1p
15A

wgsize
?