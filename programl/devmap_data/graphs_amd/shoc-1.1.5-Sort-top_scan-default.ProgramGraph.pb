

[external]
QstoreBH
F
	full_text9
7
5store i32 0, i32* @top_scan.s_seed, align 4, !tbaa !8
@callB8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
JcallBB
@
	full_text3
1
/%4 = tail call i64 @_Z12get_local_idj(i32 0) #5
2sextB*
(
	full_text

%5 = sext i32 %1 to i64
3icmpB+
)
	full_text

%6 = icmp ult i64 %4, %5
"i64B

	full_text


i64 %4
"i64B

	full_text


i64 %5
,addB%
#
	full_text

%7 = add i64 %4, 1
"i64B

	full_text


i64 %4
2icmpB*
(
	full_text

%8 = icmp eq i64 %7, %5
"i64B

	full_text


i64 %7
"i64B

	full_text


i64 %5
,andB%
#
	full_text

%9 = and i1 %6, %8
 i1B

	full_text	

i1 %6
 i1B

	full_text	

i1 %8
%brB

	full_text

br label %11
$ret8B

	full_text


ret void
Aphi8B8
6
	full_text)
'
%%12 = phi i64 [ 0, %3 ], [ %31, %30 ]
%i648B

	full_text
	
i64 %31
9br8B1
/
	full_text"
 
br i1 %6, label %15, label %13
"i18B

	full_text	

i1 %6
Xcall8BN
L
	full_text?
=
;%14 = tail call i32 @scanLocalMem(i32 0, i32* %2, i32 1) #6
'br8B

	full_text

br label %23
5mul8B,
*
	full_text

%16 = mul nsw i64 %12, %5
%i648B

	full_text
	
i64 %12
$i648B

	full_text


i64 %5
1add8B(
&
	full_text

%17 = add i64 %4, %16
$i648B

	full_text


i64 %4
%i648B

	full_text
	
i64 %16
Xgetelementptr8BE
C
	full_text6
4
2%18 = getelementptr inbounds i32, i32* %0, i64 %17
%i648B

	full_text
	
i64 %17
Hload8B>
<
	full_text/
-
+%19 = load i32, i32* %18, align 4, !tbaa !8
'i32*8B

	full_text


i32* %18
Zcall8BP
N
	full_textA
?
=%20 = tail call i32 @scanLocalMem(i32 %19, i32* %2, i32 1) #6
%i328B

	full_text
	
i32 %19
Uload8BK
I
	full_text<
:
8%21 = load i32, i32* @top_scan.s_seed, align 4, !tbaa !8
2add8B)
'
	full_text

%22 = add i32 %21, %20
%i328B

	full_text
	
i32 %21
%i328B

	full_text
	
i32 %20
Hstore8B=
;
	full_text.
,
*store i32 %22, i32* %18, align 4, !tbaa !8
%i328B

	full_text
	
i32 %22
'i32*8B

	full_text


i32* %18
'br8B

	full_text

br label %23
Dphi8B;
9
	full_text,
*
(%24 = phi i32 [ %20, %15 ], [ %14, %13 ]
%i328B

	full_text
	
i32 %20
%i328B

	full_text
	
i32 %14
Bphi8B9
7
	full_text*
(
&%25 = phi i32 [ %19, %15 ], [ 0, %13 ]
%i328B

	full_text
	
i32 %19
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
9br8B1
/
	full_text"
 
br i1 %9, label %26, label %30
"i18B

	full_text	

i1 %9
2add8B)
'
	full_text

%27 = add i32 %25, %24
%i328B

	full_text
	
i32 %25
%i328B

	full_text
	
i32 %24
Uload8BK
I
	full_text<
:
8%28 = load i32, i32* @top_scan.s_seed, align 4, !tbaa !8
2add8B)
'
	full_text

%29 = add i32 %27, %28
%i328B

	full_text
	
i32 %27
%i328B

	full_text
	
i32 %28
Ustore8BJ
H
	full_text;
9
7store i32 %29, i32* @top_scan.s_seed, align 4, !tbaa !8
%i328B

	full_text
	
i32 %29
'br8B

	full_text

br label %30
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #4
8add8B/
-
	full_text 

%31 = add nuw nsw i64 %12, 1
%i648B

	full_text
	
i64 %12
6icmp8B,
*
	full_text

%32 = icmp eq i64 %31, 16
%i648B

	full_text
	
i64 %31
:br8B2
0
	full_text#
!
br i1 %32, label %10, label %11
#i18B

	full_text


i1 %32
&i32*8B

	full_text
	
i32* %0
$i328B

	full_text


i32 %1
&i32*8B
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
-; undefined function B

	full_text

 
#i648B

	full_text	

i64 1
$i648B

	full_text


i64 16
#i328B

	full_text	

i32 0
ai32*8BU
S
	full_textF
D
B@top_scan.s_seed = internal unnamed_addr global i32 undef, align 4
#i648B

	full_text	

i64 0
#i328B

	full_text	

i32 1       	  
 
 

                 !    "# "" $$ %& %' %% () (* (( +- ,. ,, /0 // 11 23 25 46 44 77 89 8: 88 ;< ;; => ?@ ?? AB AA CD CE F G G "   	   
 ?        !  #$ &" '% ) *" - .  0 3/ 5, 64 97 :8 < @? BA D   + , ,2 42 >= >C C  HH  II JJ JJ  HH " JJ " II 1 HH 1> HH >K K ?L AM M M M /N N $N 7N ;O P P P "P 1P >"

top_scan"
_Z7barrierj"
_Z12get_local_idj"
scanLocalMem*?
shoc-1.1.5-Sort-top_scan.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?
 
transfer_bytes_log1p
? tA

transfer_bytes
???

devmap_label


wgsize_log1p
? tA

wgsize
?