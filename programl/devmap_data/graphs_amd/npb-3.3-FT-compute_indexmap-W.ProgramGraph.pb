

[external]
KcallBC
A
	full_text4
2
0%6 = tail call i64 @_Z13get_global_idj(i32 2) #2
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
KcallBC
A
	full_text4
2
0%8 = tail call i64 @_Z13get_global_idj(i32 1) #2
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
LcallBD
B
	full_text5
3
1%10 = tail call i64 @_Z13get_global_idj(i32 0) #2
6truncB-
+
	full_text

%11 = trunc i64 %10 to i32
#i64B

	full_text
	
i64 %10
4icmpB,
*
	full_text

%12 = icmp slt i32 %7, %3
"i32B

	full_text


i32 %7
4icmpB,
*
	full_text

%13 = icmp slt i32 %9, %2
"i32B

	full_text


i32 %9
/andB(
&
	full_text

%14 = and i1 %12, %13
!i1B

	full_text


i1 %12
!i1B

	full_text


i1 %13
5icmpB-
+
	full_text

%15 = icmp slt i32 %11, %1
#i32B

	full_text
	
i32 %11
/andB(
&
	full_text

%16 = and i1 %14, %15
!i1B

	full_text


i1 %14
!i1B

	full_text


i1 %15
8brB2
0
	full_text#
!
br i1 %16, label %17, label %42
!i1B

	full_text


i1 %16
4add8B+
)
	full_text

%18 = add nsw i32 %7, 16
$i328B

	full_text


i32 %7
3srem8B)
'
	full_text

%19 = srem i32 %18, 32
%i328B

	full_text
	
i32 %18
6add8B-
+
	full_text

%20 = add nsw i32 %19, -16
%i328B

	full_text
	
i32 %19
6mul8B-
+
	full_text

%21 = mul nsw i32 %20, %20
%i328B

	full_text
	
i32 %20
%i328B

	full_text
	
i32 %20
4add8B+
)
	full_text

%22 = add nsw i32 %9, 64
$i328B

	full_text


i32 %9
4srem8B*
(
	full_text

%23 = srem i32 %22, 128
%i328B

	full_text
	
i32 %22
6add8B-
+
	full_text

%24 = add nsw i32 %23, -64
%i328B

	full_text
	
i32 %23
6mul8B-
+
	full_text

%25 = mul nsw i32 %24, %24
%i328B

	full_text
	
i32 %24
%i328B

	full_text
	
i32 %24
:add8B1
/
	full_text"
 
%26 = add nuw nsw i32 %25, %21
%i328B

	full_text
	
i32 %25
%i328B

	full_text
	
i32 %21
5add8B,
*
	full_text

%27 = add nsw i32 %11, 64
%i328B

	full_text
	
i32 %11
4srem8B*
(
	full_text

%28 = srem i32 %27, 128
%i328B

	full_text
	
i32 %27
6add8B-
+
	full_text

%29 = add nsw i32 %28, -64
%i328B

	full_text
	
i32 %28
6mul8B-
+
	full_text

%30 = mul nsw i32 %29, %29
%i328B

	full_text
	
i32 %29
%i328B

	full_text
	
i32 %29
:add8B1
/
	full_text"
 
%31 = add nuw nsw i32 %26, %30
%i328B

	full_text
	
i32 %26
%i328B

	full_text
	
i32 %30
=sitofp8B1
/
	full_text"
 
%32 = sitofp i32 %31 to double
%i328B

	full_text
	
i32 %31
6fmul8B,
*
	full_text

%33 = fmul double %32, %4
+double8B

	full_text


double %32
Kcall8BA
?
	full_text2
0
.%34 = tail call double @_Z3expd(double %33) #2
+double8B

	full_text


double %33
4mul8B+
)
	full_text

%35 = mul nsw i32 %7, %2
$i328B

	full_text


i32 %7
3add8B*
(
	full_text

%36 = add nsw i32 %1, 1
1add8B(
&
	full_text

%37 = add i32 %35, %9
%i328B

	full_text
	
i32 %35
$i328B

	full_text


i32 %9
2mul8B)
'
	full_text

%38 = mul i32 %37, %36
%i328B

	full_text
	
i32 %37
%i328B

	full_text
	
i32 %36
6add8B-
+
	full_text

%39 = add nsw i32 %38, %11
%i328B

	full_text
	
i32 %38
%i328B

	full_text
	
i32 %11
6sext8B,
*
	full_text

%40 = sext i32 %39 to i64
%i328B

	full_text
	
i32 %39
^getelementptr8BK
I
	full_text<
:
8%41 = getelementptr inbounds double, double* %0, i64 %40
%i648B

	full_text
	
i64 %40
Nstore8BC
A
	full_text4
2
0store double %34, double* %41, align 8, !tbaa !8
+double8B

	full_text


double %34
-double*8B

	full_text

double* %41
'br8B

	full_text

br label %42
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %2
*double8B

	full_text

	double %4
$i328B

	full_text


i32 %3
$i328B

	full_text


i32 %1
,double*8B

	full_text


double* %0
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
$i328B

	full_text


i32 64
%i328B

	full_text
	
i32 -16
%i328B

	full_text
	
i32 -64
$i328B

	full_text


i32 16
#i328B

	full_text	

i32 2
#i328B

	full_text	

i32 1
%i328B

	full_text
	
i32 128
#i328B

	full_text	

i32 0
$i328B

	full_text


i32 32       	  
 

                      !" !! #$ ## %& %% '( ') '' *+ *, ** -. -- /0 // 12 11 34 35 33 67 68 66 9: 99 ;< ;; => == ?@ ?? AA BC BD BB EF EG EE HI HJ HH KL KK MN MM OP OQ OO RT T ?U ;V 
W W AX M   	  
             "! $# &% (% )' + , .- 0/ 21 41 5* 73 86 :9 <; > @? C DB FA GE I JH LK N= PM Q  SR S S ZZ YY= ZZ = YY  YY  YY [ ![ -\ ] %] 1^ _ ` ` Aa #a /b c "
compute_indexmap"
_Z13get_global_idj"	
_Z3expd*?
npb-FT-compute_indexmap.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282?

wgsize
@

devmap_label


wgsize_log1p
???A

transfer_bytes
???

 
transfer_bytes_log1p
???A